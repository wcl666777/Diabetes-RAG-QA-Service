"""RAG 服务基础冒烟测试（无需 Milvus / 模型 / LLM 即可运行）。

运行：pytest tests/test_health.py
"""
import httpx
import pytest

from app.config import get_settings
from app.main import app


def _auth_headers() -> dict:
    """受保护接口所需的鉴权头（未配置密钥时返回空，交由鉴权中间件拦截）。"""
    key = get_settings().rag_api_key
    return {"X-API-Key": key} if key else {}


@pytest.mark.asyncio
async def test_health():
    """健康检查应返回 ok（/health 不鉴权）。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "rag_service"


@pytest.mark.asyncio
async def test_protected_endpoint_requires_api_key():
    """未携带 API Key 访问业务接口应被拦截（401；未配置密钥时为 503）。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/v1/rag/query", json={"question": "出院后能吃什么？"})
    assert r.status_code in (401, 503)


@pytest.mark.asyncio
async def test_query_missing_question_rejected():
    """携带正确 API Key 后，缺少 question 字段应被 Pydantic 校验拦截（422）。"""
    if not get_settings().rag_api_key:
        pytest.skip("未配置 RAG_API_KEY，跳过该用例")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/v1/rag/query", json={}, headers=_auth_headers())
    assert r.status_code == 422