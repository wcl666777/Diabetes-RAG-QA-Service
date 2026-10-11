"""BGE-Reranker 精排模块（进程内单例）。

Cross-Encoder 精排：对「查询-文档」对打分，比双塔向量检索更精准。
用法：rerank(query, docs) -> [(doc, score), ...] 按分数降序。
"""
import math
from typing import List

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_instance = None


def _load_model():
    from sentence_transformers import CrossEncoder

    path = get_settings().reranker_model_path
    logger.info("reranker.loading | path=%s", path)
    # 本机为 CPU 环境（无 GPU），用 fp32 推理（bge-reranker 权重为 cross-encoder 格式）
    return CrossEncoder(path, device="cpu")


def get_reranker():
    global _instance
    if _instance is None:
        _instance = _load_model()
        logger.info("reranker.loaded")
    return _instance


def rerank_pairs(query: str, docs: List[str]) -> List[float]:
    """对 query 与每个 doc 计算相关性分数，返回与 docs 等长的分数列表。

    CrossEncoder 输出原始 logits，这里用 sigmoid 归一化到 [0,1]，
    与原 FlagReranker.compute_score(normalize=True) 语义保持一致。
    """
    if not docs:
        return []
    reranker = get_reranker()
    pairs = [[query, doc] for doc in docs]
    logits = reranker.predict(pairs, show_progress_bar=False)
    return [1.0 / (1.0 + math.exp(-s)) for s in logits]