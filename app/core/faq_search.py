"""高频问答快车道。

作用：在进入 RAG 之前，先用 BM25 在「标准问答库」里找最相近的问题。
- 若相似度超过阈值 → 直接返回库里预置的标准答案（快、稳、可控）；
- 否则 → 返回 need_rag=True，交给 RAG 检索生成。

语料的存取：优先走 Redis 缓存（标准问题列表 + 分词结果），缓存缺失时回源 MySQL，
分词后回写 Redis，避免每次启动都重新分词。
"""
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.core import cache
from app.core import mysql_client as mysql_module
from app.core.logger import get_logger
from app.core.preprocess import preprocess_text

logger = get_logger(__name__)

# Redis 缓存键
_KEY_QUESTIONS = "qa_original_questions"
_KEY_TOKENIZED = "qa_tokenized_questions"


class FaqSearch:
    """高频问答快车道检索器。"""

    def __init__(self) -> None:
        self.mysql_client = mysql_module.get_mysql_client()
        self.questions: List[str] = []
        self.tokenized: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        # MySQL 不可用则快车道整体降级（全部交给 RAG）
        if self.mysql_client is not None:
            self._load_data()
        else:
            logger.warning("faq.disabled | mysql unavailable")

    def _load_data(self) -> None:
        """加载标准问题语料：Redis 优先，缺失则回源 MySQL 并写回缓存。"""
        self.questions = cache.get_json(_KEY_QUESTIONS) or []
        self.tokenized = cache.get_json(_KEY_TOKENIZED) or []

        if not self.questions:
            self.questions = self.mysql_client.fetch_questions()
            cache.set_json(_KEY_QUESTIONS, self.questions)
        if not self.tokenized and self.questions:
            self.tokenized = [preprocess_text(q) for q in self.questions]
            cache.set_json(_KEY_TOKENIZED, self.tokenized)

        if self.tokenized:
            self.bm25 = BM25Okapi(self.tokenized)
        logger.info("faq.loaded | questions=%d", len(self.questions))

    def reload(self) -> None:
        """问答库更新后重新加载（清缓存由调用方负责）。"""
        cache.set_json(_KEY_QUESTIONS, [])
        cache.set_json(_KEY_TOKENIZED, [])
        self._load_data()

    def search(self, query: str, threshold: Optional[float] = None) -> Tuple[Optional[str], bool]:
        """快车道检索。

        Returns:
            (answer, need_rag)
            - 命中： answer=标准答案, need_rag=False
            - 未命中：answer=None,   need_rag=True（交给 RAG）
        """
        s = get_settings()
        if not s.faq_enabled or self.mysql_client is None:
            return None, True
        threshold = threshold if threshold is not None else s.faq_hit_threshold

        # ① 先查答案缓存
        cached = cache.get_answer(query)
        if cached:
            return cached, False

        # ② 无语料 → 直接走 RAG
        if not self.bm25 or not self.questions:
            return None, True

        # ③ BM25 打分：以「最佳标准问题得分 ÷ query 自身满分」作为相似度。
        #    不能对 BM25 分数做 softmax：softmax 会把最大分放大到接近 1，
        #    与是否真的相似无关（实测完全相同的问题仅 0.67、语义无关的问题却 0.93）。
        #    把 query 自身作为伪文档加入语料，其得分即「完美匹配」的理论上限。
        q_tokens = preprocess_text(query)
        bm25 = BM25Okapi(self.tokenized + [q_tokens])
        scores = bm25.get_scores(q_tokens)
        self_score = scores[-1]
        corpus_scores = scores[:-1]
        if self_score <= 0 or len(corpus_scores) == 0:
            return None, True

        best_idx = max(range(len(corpus_scores)), key=lambda i: corpus_scores[i])
        ratio = corpus_scores[best_idx] / self_score

        if ratio > threshold:
            answer = self.mysql_client.fetch_answer(self.questions[best_idx])
            if answer:
                # 命中即缓存「问题 → 答案」，下次同问题直接秒回
                cache.set_answer(query, answer)
                logger.info("faq.hit | ratio=%.3f | question=%s", ratio, query[:30])
                return answer, False

        logger.info("faq.miss | ratio=%.3f | question=%s", ratio, query[:30])
        return None, True

    def close(self) -> None:
        self.mysql_client.close()


_faq_instance: Optional["FaqSearch"] = None


def get_faq() -> "FaqSearch":
    """高频问答快车道单例。

    REST 接口与 MCP 工具共享同一实例，避免重复构建 BM25 语料与重复占内存。
    """
    global _faq_instance
    if _faq_instance is None:
        _faq_instance = FaqSearch()
    return _faq_instance
