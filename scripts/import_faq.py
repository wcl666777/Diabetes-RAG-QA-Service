"""导入高频问答对到 MySQL（faq_qa 表），启用 RAG 前置「快车道」。

做法：
把「标准问题 → 标准答案」预先写入 MySQL，问答时先用 BM25 匹配，命中即直答。

支持两种文件，均可用 2 列或 3 列（列顺序/中英文表头都可）：
  1) CSV：逗号分隔，首行表头
         分类,问题,答案            ← 3 列（分类可省略）
         问题,答案 / question,answer ← 2 列，分类自动填「通用」
  2) TXT：竖线分隔，每行一条
         分类|问题|答案   或   问题|答案
     （首行若为表头会自动跳过）

用法（在 rag_service 目录执行）：
    python scripts/import_faq.py --file ./data/faq_qa.csv            # 追加导入
    python scripts/import_faq.py --file ./data/faq_qa.csv --truncate # 先清空再导入

说明：
  - 默认「追加」；反复导入会产生重复，建议加 --truncate 保持表内数据干净。
  - 导入后会清空 Redis 里的快车道语料缓存；但服务启动时已预热过语料，
    需重启 RAG 服务（uvicorn）后新数据才生效。
"""
import argparse
import csv
import sys
from pathlib import Path

# 保证以 `python scripts/import_faq.py` 运行时能导入 app 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.mysql_client import MysqlClient

# 表头别名：中文 / 英文都认，大小写不敏感
_HEADER_ALIASES = {
    "分类": "category", "category": "category", "科室": "category", "类别": "category",
    "问题": "question", "question": "question",
    "答案": "answer", "answer": "answer",
}

_DEFAULT_CATEGORY = "通用"
_KEY_QUESTIONS = "qa_original_questions"
_KEY_TOKENIZED = "qa_tokenized_questions"


def _map_header(cells) -> dict:
    """把表头行映射为 {字段: 列号}；question/answer 必须齐，否则返回 None。"""
    mapping = {}
    for i, c in enumerate(cells):
        key = _HEADER_ALIASES.get(c.strip().lower())
        if key and key not in mapping:
            mapping[key] = i
    return mapping if "question" in mapping and "answer" in mapping else None


def _cell(row, idx) -> str:
    if idx is None or idx >= len(row):
        return ""
    return row[idx].strip()


def _to_rows(records) -> list:
    """把「行列表」规整为 [{category, question, answer}]，自动识别表头与列数。"""
    if not records:
        return []
    mapping = _map_header(records[0])
    data = records[1:] if mapping else records
    if not mapping:
        # 无表头：按列数推断（>=3 列视为 分类/问题/答案，否则 问题/答案）
        width = max((len(r) for r in data), default=0)
        mapping = (
            {"category": 0, "question": 1, "answer": 2}
            if width >= 3
            else {"question": 0, "answer": 1}
        )
    rows = []
    for r in data:
        question = _cell(r, mapping.get("question"))
        answer = _cell(r, mapping.get("answer"))
        if not question or not answer:
            continue
        rows.append({
            "category": _cell(r, mapping.get("category")) or _DEFAULT_CATEGORY,
            "question": question,
            "answer": answer,
        })
    return rows


def _read_csv(path: Path) -> list:
    """解析 CSV（自动跳过空行，兼容 Excel 导出的 BOM）。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        records = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    return _to_rows(records)


def _read_txt(path: Path) -> list:
    """解析 TXT：每行用竖线分隔；答案中的竖线会原样保留。"""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []
    # 先按竖线切分，判断首行是否表头
    first = lines[0].split("|")
    has_header = _map_header(first) is not None
    data = lines[1:] if has_header else lines
    width = max((len(ln.split("|")) for ln in data), default=0)

    rows = []
    for ln in data:
        parts = ln.split("|")
        if width >= 3:
            category, question, answer = parts[0].strip(), _cell(parts, 1), "|".join(parts[2:]).strip()
        else:
            category, question, answer = _DEFAULT_CATEGORY, _cell(parts, 0), "|".join(parts[1:]).strip()
        if question and answer:
            rows.append({"category": category or _DEFAULT_CATEGORY, "question": question, "answer": answer})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="导入高频问答对到 MySQL faq_qa 表")
    parser.add_argument("--file", required=True, help="CSV 或 TXT 文件路径")
    parser.add_argument("--truncate", action="store_true", help="导入前清空 faq_qa 表（避免重复）")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[错误] 文件不存在：{path}")
        sys.exit(1)

    rows = _read_txt(path) if path.suffix.lower() == ".txt" else _read_csv(path)
    if not rows:
        print("[错误] 未解析到有效数据，请确认文件为「问题/答案」或「分类/问题/答案」")
        sys.exit(1)
    print(f"[解析] {path.name} → {len(rows)} 条问答对")

    client = MysqlClient()
    client.create_table()
    if args.truncate:
        client.cursor.execute(f"TRUNCATE TABLE {client.table}")
        client.connection.commit()
        print("[清空] faq_qa 表已清空")

    inserted = client.insert_faq(rows)
    total = len(client.fetch_questions())
    client.close()

    # 清空 Redis 快车道语料缓存（下次加载会回源 MySQL 重新分词）
    try:
        from app.core import cache
        cache.set_json(_KEY_QUESTIONS, [])
        cache.set_json(_KEY_TOKENIZED, [])
    except Exception:
        pass

    print(f"[完成] 本次导入 {inserted} 条，faq_qa 当前共 {total} 条")
    print("[提示] 请重启 RAG 服务，快车道才会加载新语料。")


if __name__ == "__main__":
    main()
