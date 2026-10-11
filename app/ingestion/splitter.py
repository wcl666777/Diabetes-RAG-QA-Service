"""中文感知的父子块分块。

父子块设计：
- 父块（大，默认 1200 字）：作为完整上下文，命中子块后回溯取用；
- 子块（小，默认 300 字）：真正入库、参与向量/关键词检索的最小单位。

只把「子块」写入 Milvus，但每条子块记录里冗余存一份 parent_content，
这样检索命中子块后可原地回溯父块，无需二次查询。
"""
import hashlib
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# 针对中文优化的分隔符
_ZH_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ".", "!", "?", ";", " ", ""]


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_ZH_SEPARATORS,
        length_function=len,
    )


def _chunk_id(parent_id: str, k: int, content: str) -> str:
    """子块唯一 ID：内容不变则 ID 稳定，支持 upsert 幂等。"""
    raw = f"{parent_id}_{k}_{content[:50]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def split_parent_child(docs: List[dict]) -> List[dict]:
    """把文档列表切分为子块列表（每个子块携带父块信息）。

    docs: [{"content": str, "source": str}]（loader 的输出）
    return: [{"id","content","parent_id","parent_content","source"}]
    """
    s = get_settings()
    parent_splitter = _make_splitter(s.parent_chunk_size, s.chunk_overlap)
    child_splitter = _make_splitter(s.child_chunk_size, s.chunk_overlap)

    children: List[dict] = []
    for i, doc in enumerate(docs):
        source = doc.get("source", "unknown")

        # 父块切分：可选「模型语义分句 + 按长度组装」，默认用中文递归切分
        if s.use_semantic_splitter:
            try:
                from app.ingestion.semantic_splitter import merge_to_chunks, split_sentences
                sentences = split_sentences(doc["content"], s.semantic_splitter_model_path)
                parent_texts = merge_to_chunks(sentences, s.parent_chunk_size)
            except Exception as e:
                logger.warning("splitter.semantic_failed | fallback=recursive | error=%s", e)
                parent_texts = parent_splitter.split_text(doc["content"])
        else:
            parent_texts = parent_splitter.split_text(doc["content"])

        for j, parent_text in enumerate(parent_texts):
            parent_text = parent_text.strip()
            if not parent_text:
                continue
            parent_id = f"doc_{i}_parent_{j}"
            for k, child_text in enumerate(child_splitter.split_text(parent_text)):
                child_text = child_text.strip()
                if not child_text:
                    continue
                children.append(
                    {
                        "id": _chunk_id(parent_id, k, child_text),
                        "content": child_text,
                        "parent_id": parent_id,
                        "parent_content": parent_text,
                        "source": source,
                    }
                )

    logger.info("splitter.done | docs=%d | children=%d", len(docs), len(children))
    return children
