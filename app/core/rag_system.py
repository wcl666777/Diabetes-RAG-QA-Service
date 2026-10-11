"""RAG 主流程。

主线：查询分类 → 检索策略选择 → 检索（4 种策略）→ 重排 → 生成。

- 查询分类：BERT 判「通用知识 / 专业咨询」；通用知识直接由大模型回答，不检索；
- 专业咨询：LLM 从「直接 / HyDE / 子查询 / 回溯」中选一种策略去检索；
- 检索统一走 retriever.retrieve（Milvus 稠密+稀疏混合检索 + 父块回溯 + 精排）；
- 生成统一走 prompts.rag_prompt（含历史 + 上下文 + 客服电话）。

说明：为避免「策略路径误用流式生成器」的问题，本实现里
策略类子调用（HyDE / 子查询 / 回溯）一律用非流式 llm.generate；仅最终回答支持流式。
"""
from typing import Iterator, List, Optional, Tuple

from app.config import get_settings
from app.core import llm, retriever
from app.core.logger import get_logger
from app.core.prompts import (
    backtracking_prompt,
    hyde_prompt,
    rag_prompt,
    subquery_prompt,
)
from app.core.strategy_selector import (
    STRATEGY_BACKTRACKING,
    STRATEGY_DIRECT,
    STRATEGY_HYDE,
    STRATEGY_SUBQUERY,
    StrategySelector,
)

logger = get_logger(__name__)

# 查询类别
CATEGORY_GENERAL = "通用知识"
CATEGORY_SPECIALIZED = "专业咨询"

_MAX_SUBQUERIES = 5   # 子查询最多拆几条，防止过多检索拖慢

_instance: Optional["RAGSystem"] = None


class RAGSystem:
    """RAG 主流程编排器。"""

    def __init__(self) -> None:
        s = get_settings()
        self.top_k = s.rerank_top_k
        self.candidate_m = s.candidate_m
        self.phone = s.service_phone
        self._classifier = None
        self._selector = StrategySelector()

    # ── 查询分类（延迟加载 BERT，避免非必需时占用内存）──
    def _classify(self, query: str) -> str:
        s = get_settings()
        if not s.enable_query_classifier:
            return CATEGORY_SPECIALIZED
        if self._classifier is None:
            from app.core.query_classifier import get_classifier
            self._classifier = get_classifier()
        return self._classifier.predict_category(query)

    # ── 4 种检索策略 ──
    def _retrieve_with_hyde(self, query: str) -> Tuple[List[dict], float]:
        """假设问题检索：先让 LLM 生成假设答案，再用假设答案检索。"""
        try:
            hypo = llm.generate(hyde_prompt().format(query=query))
            if hypo.strip():
                return retriever.retrieve(hypo.strip(), top_k=self.top_k)
        except Exception as e:
            logger.warning("rag.hyde_failed | error=%s", e)
        return retriever.retrieve(query, top_k=self.top_k)

    def _retrieve_with_subqueries(self, query: str) -> Tuple[List[dict], float]:
        """子查询检索：拆成多个子查询分别检索，再按内容去重合并。"""
        try:
            text = llm.generate(subquery_prompt().format(query=query))
            subs = [ln.strip() for ln in text.split("\n") if ln.strip()][:_MAX_SUBQUERIES]
        except Exception as e:
            logger.warning("rag.subquery_failed | error=%s", e)
            subs = []

        if not subs:
            return retriever.retrieve(query, top_k=self.top_k)

        merged: List[dict] = []
        seen = set()
        best_conf = 0.0
        for sub in subs:
            docs, conf = retriever.retrieve(sub, top_k=self.top_k)
            best_conf = max(best_conf, conf)
            for d in docs:
                key = d.get("parent_id") or d["content"]
                if key not in seen:
                    seen.add(key)
                    merged.append(d)
        return merged, best_conf

    def _retrieve_with_backtracking(self, query: str) -> Tuple[List[dict], float]:
        """回溯问题检索：先把复杂问题简化，再检索。"""
        try:
            simple = llm.generate(backtracking_prompt().format(query=query))
            if simple.strip():
                return retriever.retrieve(simple.strip(), top_k=self.top_k)
        except Exception as e:
            logger.warning("rag.backtracking_failed | error=%s", e)
        return retriever.retrieve(query, top_k=self.top_k)

    def retrieve_and_merge(self, query: str, strategy: Optional[str] = None) -> Tuple[List[dict], float, str]:
        """按策略检索并合并，返回 (父块文档, confidence, 实际策略)。"""
        if strategy is None:
            strategy = self._selector.select_strategy(query)

        if strategy == STRATEGY_HYDE:
            docs, conf = self._retrieve_with_hyde(query)
        elif strategy == STRATEGY_SUBQUERY:
            docs, conf = self._retrieve_with_subqueries(query)
        elif strategy == STRATEGY_BACKTRACKING:
            docs, conf = self._retrieve_with_backtracking(query)
        else:
            strategy = STRATEGY_DIRECT
            docs, conf = retriever.retrieve(query, top_k=self.top_k)

        # 空召回兜底：回退直接检索
        if not docs:
            strategy = STRATEGY_DIRECT
            docs, conf = retriever.retrieve(query, top_k=self.top_k)

        return docs[: self.candidate_m], conf, strategy

    # ── 上下文与 Prompt 组装 ──
    @staticmethod
    def _format_history(history: Optional[List[dict]]) -> str:
        if not history:
            return "（无）"
        return "\n".join(
            f"问：{h.get('question')}\n答：{h.get('answer')}" for h in history[-5:]
        )

    def _build_context(self, docs: List[dict]) -> str:
        if not docs:
            return "（未检索到相关资料）"
        return "\n\n".join(f"【来源：{d.get('source', '')}】\n{d['content']}" for d in docs)

    def prepare(self, query: str, history: Optional[List[dict]] = None) -> dict:
        """预处理：分类 → （专业）策略检索 → 组装 Prompt。

        Returns dict(query_category, strategy, docs, confidence, sources, prompt)
        """
        category = self._classify(query)
        strategy = "无（通用知识）"
        docs: List[dict] = []
        confidence = 0.0

        if category == CATEGORY_SPECIALIZED:
            docs, confidence, strategy = self.retrieve_and_merge(query)
            context = self._build_context(docs)
        else:
            logger.info("rag.general_knowledge | query=%s", query[:30])
            # 通用交流类问题：不检索，直接让模型用自己的知识友好回答
            context = "（本次为日常交流，无需参考资料，请直接用你自己的知识友好回答）"

        prompt = rag_prompt().format(
            context=context,
            history=self._format_history(history),
            question=query,
            phone=self.phone,
        )
        return {
            "query_category": category,
            "strategy": strategy,
            "docs": docs,
            "confidence": confidence,
            "sources": [d.get("source", "") for d in docs],
            "context": context,   # 融合后的上下文，供 MCP 工具交给 Agent 侧生成
            "prompt": prompt,
        }

    def generate_answer(self, query: str, history: Optional[List[dict]] = None) -> dict:
        """非流式生成：返回 answer / confidence / sources / 分类与策略。"""
        p = self.prepare(query, history)
        answer = llm.generate(p["prompt"])
        p["answer"] = answer
        return p

    def generate_answer_stream(
        self, query: str, history: Optional[List[dict]] = None
    ) -> Tuple[Iterator[str], dict]:
        """流式生成：返回 (token 迭代器, prepare 结果)，先给 meta 再逐 token 推送。"""
        p = self.prepare(query, history)
        return llm.generate_stream(p["prompt"]), p


def get_rag_system() -> RAGSystem:
    """RAG 系统单例。"""
    global _instance
    if _instance is None:
        _instance = RAGSystem()
    return _instance
