"""RAG 问答接口。

一次问答的三态：
  ① 命中高频问答快车道（BM25 + 阈值）→ 直接返回标准答案（快、稳）
  ② 未命中 → 交给 RAGSystem（分类 → 策略 → 检索 → 生成）
  ③ 无任何结果 → 兜底话术

接口：
  POST   /api/v1/rag/query                 非流式问答
  POST   /api/v1/rag/query/stream          SSE 流式问答
  POST   /api/v1/rag/session               创建会话 ID
  GET    /api/v1/rag/history/{session_id}  读取会话历史
  DELETE /api/v1/rag/history/{session_id}  清除会话历史
"""
import asyncio
import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.core import conversation
from app.core.logger import get_logger
from app.core.rag_system import get_rag_system

router = APIRouter(prefix="/api/v1/rag", tags=["RAG 问答"])
logger = get_logger(__name__)


def _get_faq():
    """高频问答快车道单例（延迟初始化；与 MCP 工具共享同一实例，MySQL 不可用则内部降级）。"""
    from app.core.faq_search import get_faq

    return get_faq()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="会话 ID，缺省自动生成")
    source_filter: Optional[str] = Field(None, description="来源过滤，如指南文件名")
    history: Optional[List[dict]] = Field(None, description="最近对话历史 [{question, answer}]")
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    session_id: str
    source: str = "rag"                     # faq / rag / none：答案来自哪条路径
    query_category: str = ""                # 通用知识 / 专业咨询
    strategy: str = ""                      # 检索策略
    is_streaming: bool = False


def _resolve_history(req: QueryRequest) -> tuple[List[dict], str]:
    """确定会话 ID 与历史：优先用请求传入的 history，否则读库。"""
    session_id = req.session_id or str(uuid.uuid4())
    history = req.history if req.history else conversation.get_session_history(session_id)
    return history, session_id


@router.post("/session")
async def create_session():
    """创建会话 ID（前端进入问答时调用）。"""
    return {"session_id": str(uuid.uuid4())}


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    """读取指定会话的最近历史。"""
    return {"session_id": session_id, "history": conversation.get_session_history(session_id)}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """清除指定会话历史。"""
    ok = conversation.clear_session_history(session_id)
    return {"status": "ok" if ok else "failed", "session_id": session_id}


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """非流式问答：快车道 → RAG → 兜底。"""
    history, session_id = _resolve_history(req)

    # ① 高频问答快车道
    answer, need_rag = _get_faq().search(req.question)
    if answer:
        conversation.update_session_history(session_id, req.question, answer)
        return QueryResponse(
            answer=answer, confidence=1.0, sources=[], session_id=session_id, source="faq",
        )

    # ② RAG 生成
    try:
        result = get_rag_system().generate_answer(req.question, history)
    except Exception as e:
        logger.error("qa.query_failed | error=%s", e)
        raise HTTPException(status_code=502, detail=f"生成失败：{e}")

    conversation.update_session_history(session_id, req.question, result["answer"])
    return QueryResponse(
        answer=result["answer"],
        confidence=round(result["confidence"], 4),
        sources=result["sources"],
        session_id=session_id,
        source="rag" if result["docs"] else "none",
        query_category=result["query_category"],
        strategy=result["strategy"],
    )


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE 流式问答：快车道命中则一次性推送；否则先发 meta 再逐 token 推送。

    事件类型：meta（分类/策略/置信度/来源）、token（增量）、done、error
    """
    history, session_id = _resolve_history(req)

    async def event_generator():
        # ① 快车道
        answer, need_rag = _get_faq().search(req.question)
        if answer:
            yield _sse({"type": "meta", "source": "faq", "confidence": 1.0, "sources": []})
            yield _sse({"type": "token", "content": answer})
            conversation.update_session_history(session_id, req.question, answer)
            yield _sse({"type": "done", "session_id": session_id})
            return

        # ② RAG：先检索 + 组装 prompt（同步 CPU/IO，放线程池）
        loop = asyncio.get_running_loop()
        try:
            sys = get_rag_system()
            generator, p = await loop.run_in_executor(
                None, lambda: sys.generate_answer_stream(req.question, history)
            )
        except Exception as e:
            logger.error("qa.stream_prepare_failed | error=%s", e)
            yield _sse({"type": "error", "message": str(e)})
            return

        yield _sse({
            "type": "meta",
            "source": "rag" if p["docs"] else "none",
            "confidence": round(p["confidence"], 4),
            "sources": p["sources"],
            "query_category": p["query_category"],
            "strategy": p["strategy"],
            "session_id": session_id,
        })

        parts = []
        try:
            for token in generator:
                parts.append(token)
                yield _sse({"type": "token", "content": token})
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error("qa.stream_failed | error=%s", e)
            yield _sse({"type": "error", "message": str(e)})
            return

        conversation.update_session_history(session_id, req.question, "".join(parts))
        yield _sse({"type": "done", "session_id": session_id})

    return EventSourceResponse(event_generator())


def _sse(payload: dict) -> dict:
    """封装 SSE data 行。"""
    return {"data": json.dumps(payload, ensure_ascii=False)}
