"""知识库构建：加载 → 父子块切分 → 稠密/稀疏向量化 → 写入 Milvus。

用法：
    python -m app.ingestion.build --dir ./data
或：
    from app.ingestion.build import build_knowledge_base; build_knowledge_base("./data")

说明：知识库检索已由「独立 BM25 语料」改为「Milvus 内稠密+稀疏混合检索」，
因此这里不再重建 BM25 语料。
"""
import argparse

from app.core import embedder, milvus_store
from app.core.logger import get_logger
from app.ingestion.loader import load_directory
from app.ingestion.splitter import split_parent_child

logger = get_logger(__name__)


def build_knowledge_base(dir_path: str) -> int:
    """执行完整入库流程，返回写入的子块数。"""
    logger.info("build_kb.start | dir=%s", dir_path)

    docs = load_directory(dir_path)
    if not docs:
        logger.warning("build_kb.no_docs")
        return 0

    # ① 父子块切分
    children = split_parent_child(docs)
    if not children:
        logger.warning("build_kb.no_chunks")
        return 0
    logger.info("build_kb.splitted | children=%d", len(children))

    # ② 稠密 + 稀疏向量化（批量）
    dense_list, sparse_list = embedder.encode_documents([c["content"] for c in children])

    # ③ 组装写库数据（每条子块带上父块内容）
    rows = []
    for child, dense, sparse in zip(children, dense_list, sparse_list):
        rows.append(
            {
                "id": child["id"],
                "content": child["content"],
                "parent_id": child["parent_id"],
                "parent_content": child["parent_content"],
                "source": child["source"],
                "dense": dense,
                "sparse": sparse,
            }
        )

    milvus_store.insert_chunks(rows)
    logger.info("build_kb.done | children=%d", len(children))
    return len(children)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建医学指南知识库")
    parser.add_argument("--dir", default="./data", help="文档目录")
    args = parser.parse_args()
    n = build_knowledge_base(args.dir)
    print(f"知识库构建完成，共写入 {n} 个子块。")
    print(f"当前向量库文档数：{milvus_store.count_docs()}")
