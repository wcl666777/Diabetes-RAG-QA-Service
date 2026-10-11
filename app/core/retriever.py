"""混合检索 + 重排 主流程。

流程：query → BGE-M3 稠密/稀疏编码 → Milvus 混合检索(WeightedRanker 融合)
      → 按父块去重（命中子块回溯父块）→ bge-reranker 精排 → 返回 Top-M 父块。

对外主入口：retrieve(query, top_k) -> (ranked_docs, confidence)
    ranked_docs: [{"content","source","parent_id","score"}]，content 为父块完整内容
    confidence:  精排首条分数（供 Agent 判断是否转人工）
"""
from typing import List, Tuple

from app.config import get_settings
from app.core import embedder, milvus_store, reranker
from app.core.logger import get_logger

logger = get_logger(__name__)


def retrieve(query: str, top_k: int = 5, source_filter: str = None) -> Tuple[List[dict], float]:
    """混合检索主入口：稠密+稀疏召回 → 父块去重 → 精排。"""
    s = get_settings()

    # ① 查询编码（dense + sparse）
    q_dense, q_sparse = embedder.encode_query(query)

    # ② Milvus 混合检索（内部已按父块去重，返回父块候选）
    try:
        candidates = milvus_store.hybrid_search(
            q_dense, q_sparse, top_k=s.hybrid_candidate_k, source_filter=source_filter
        )
    except Exception as e:
        logger.warning("retrieve.hybrid_failed | error=%s", e)
        candidates = []

    if not candidates:
        return [], 0.0

    # ③ 精排（对父块内容打分）
    limit = min(len(candidates), s.hybrid_candidate_k)
    targets = candidates[:limit]
    scores = reranker.rerank_pairs(query, [c["content"] for c in targets])

    ranked = []
    for doc, score in zip(targets, scores):
        item = dict(doc)
        item["score"] = round(float(score), 6)
        ranked.append(item)
    ranked.sort(key=lambda d: d["score"], reverse=True)
    ranked = ranked[:top_k]

    confidence = ranked[0]["score"] if ranked else 0.0
    logger.info("retrieve.done | query=%s | hits=%d | conf=%.3f", query[:30], len(ranked), confidence)
    return ranked, confidence
