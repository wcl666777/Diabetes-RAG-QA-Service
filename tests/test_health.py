"""RAG 服务基础冒烟测试（无需 Milvus / 模型 / LLM 即可运行）。

运行：pytest tests/test_health.py
"""
import httpx
import pytest

from app.main import app


@pytest.mark.asyncio
async def test_health():
    """健康检查应返回 ok。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "rag_service"


@pytest.mark.asyncio
async def test_query_missing_question_rejected():
    """缺少 question 字段应被 Pydantic 校验拦截（422）。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/v1/rag/query", json={})
    assert r.status_code == 422