"""RAG 知识检索 MCP Server（供 Agent 的问答 Agent 通过 MCP 协议远程调用）。

采用 FastMCP 的 stateless_http 模式 + json_response，每次调用自包含、无需握手。
默认由 app.main 挂载到 /rag（与 REST 同进程）；如需独立进程运行，可选：
    python -m app.mcp.server     # 默认 8002，端点 http://localhost:8002/mcp
"""
import asyncio

from mcp.server.fastmcp import FastMCP

from app.core.logger import get_logger

logger = get_logger(__name__)

mcp = FastMCP(
    name="Diabetes-RAG-KnowledgeBase",
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def search_knowledge_base(
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """对糖尿病出院管理知识库执行「向量 + BM25 混合检索 + 重排」。

    Args:
        query:  中文检索问题，如「出院后能不能吃西瓜」
        top_k:  返回文档条数（默认 3）

    Returns:
        精排后的文档列表，每项含 content / source / score，
        并在最后附一条 confidence（首条精排分数，供 Agent 判断是否转人工）。
    """
    from app.core import retriever

    # 检索涉及 BGE-M3 编码 + reranker 推理（CPU-bound），放线程池避免阻塞事件循环
    loop = asyncio.get_running_loop()
    ranked, confidence = await loop.run_in_executor(
        None,
        lambda: retriever.retrieve(query, top_k=top_k),
    )

    logger.info("kb_mcp.search_done | query=%s | hits=%d | conf=%.3f",
                query[:30], len(ranked), confidence)

    return [
        {
            "content": doc["content"],
            "source": doc["source"],
            "score": doc["score"],
            "confidence": round(confidence, 4),
        }
        for doc in ranked
    ]


@mcp.tool()
async def rag_query(query: str, personal_context: str = "") -> dict:
    """完整问答备料：Redis 缓存 → MySQL 快车道 → 意图识别 → 检索策略 → 混合检索 → 融合个人资料。

    与 search_knowledge_base 的区别：本工具走 RAG 主流程，先做「通用知识 / 专业咨询」
    意图识别，再按 LLM 选出的策略检索，并由本服务统一融合患者个人资料，最后把「答案」
    或「融合好的上下文」交给 Agent。

    Args:
        query:            患者问题，如「出院后能不能吃西瓜」
        personal_context: 患者个人资料（病例 / 出院小结）文本，由 Agent 本地检索后传入

    Returns:
        {source, answer, context, confidence, query_category, strategy, sources}
        - source="faq"：命中快车道，answer 为标准答案（Agent 直接输出，无需再生成）
        - source="rag"/"none"：context 为融合好的上下文，Agent 用它流式生成
    """
    from app.core.faq_search import get_faq
    from app.core.rag_system import get_rag_system

    # 快车道 / 检索 / 策略选择涉及 BM25、BERT、LLM、BGE-M3（CPU/IO-bound），
    # 统一放线程池执行，避免阻塞事件循环。
    loop = asyncio.get_running_loop()

    def _run() -> dict:
        # ① 高频问答快车道（Redis 答案缓存 + MySQL BM25 相似问）
        answer, _need_rag = get_faq().search(query)
        if answer:
            return {
                "source": "faq",
                "answer": answer,
                "context": "",
                "confidence": 1.0,
                "query_category": "",
                "strategy": "",
                "sources": [],
            }

        # ② RAG 主流程：意图识别 → 检索策略 → 混合检索 → 组装上下文
        p = get_rag_system().prepare(query)
        context = p["context"]
        if personal_context and personal_context.strip():
            # 个人资料由本服务统一融合，避免 Agent 侧二次拼接导致口径不一致
            context = f"{context}\n\n【患者个人资料】\n{personal_context.strip()}"

        return {
            "source": "rag" if p["docs"] else "none",
            "answer": "",
            "context": context,
            "confidence": round(p["confidence"], 4),
            "query_category": p["query_category"],
            "strategy": p["strategy"],
            "sources": p["sources"],
        }

    result = await loop.run_in_executor(None, _run)
    logger.info(
        "kb_mcp.rag_query | query=%s | source=%s | cat=%s | strategy=%s | conf=%.3f",
        query[:30], result["source"], result["query_category"],
        result["strategy"], result["confidence"],
    )
    return result


if __name__ == "__main__":
    import uvicorn

    # 预热本地模型（BGE-M3 嵌入 + reranker 首次加载较慢），
    # 避免首个 MCP 请求因冷启动而触发 Agent 端调用超时。
    from app.core import embedder, reranker

    embedder.get_embedder()
    reranker.get_reranker()

    port = 8002
    print(f"RAG Knowledge Base MCP Server -> http://localhost:{port}/mcp")
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)