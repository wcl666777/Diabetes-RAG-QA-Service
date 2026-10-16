"""RAG 服务入口：FastAPI + REST 接口。

MCP Server 与 REST 同进程挂载在 /rag（共享已加载的 BGE-M3 / reranker，避免重复占用内存）：
    uvicorn app.main:app --host 0.0.0.0 --port 8001
    MCP 端点：http://host:8001/rag/mcp

如需独立进程运行，可选：python -m app.mcp.server
"""
import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import kb, qa
from app.config import get_settings
from app.core.logger import configure_logging, get_logger
from app.mcp.server import mcp as kb_mcp

settings = get_settings()
logger = get_logger(__name__)

# MCP 子应用（与 REST 同进程，共享已加载的 BGE-M3 / reranker，避免重复占用内存）
_kb_app = kb_mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level, settings.log_dir)
    logger = get_logger(__name__)
    logger.info("rag_service.starting | port=%s", settings.app_port)

    # ① 连接 Milvus 并确保集合存在（失败只告警，不拦启动）
    try:
        from app.core import milvus_store
        milvus_store.create_collection_if_not_exists()
    except Exception as e:
        logger.warning("rag_service.milvus_failed | error=%s", e)

    # ② 顺序预热本地模型（BGE-M3 + reranker；顺序加载避免并发峰值内存导致加载失败）
    try:
        from app.core import embedder, reranker
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, embedder.get_embedder)
        await loop.run_in_executor(None, reranker.get_reranker)
        logger.info("rag_service.models_warmed_up")
    except Exception as e:
        logger.warning("rag_service.model_warmup_failed | error=%s", e)

    # ③ 初始化会话历史表（MySQL 不可用时仅告警）
    try:
        from app.core import conversation
        conversation.init_conversation_table()
    except Exception as e:
        logger.warning("rag_service.conversation_init_failed | error=%s", e)

    # ④ 预热高频问答快车道（加载标准问答语料，MySQL 不可用则内部降级）
    try:
        from app.api.qa import _get_faq
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _get_faq)
        logger.info("rag_service.faq_warmed_up")
    except Exception as e:
        logger.warning("rag_service.faq_warmup_failed | error=%s", e)

    # ⑤ 驱动 MCP 子应用生命周期，与 REST 同进程对外服务
    async with _kb_app.router.lifespan_context(_kb_app):
        logger.info("rag_service.started")
        yield

    logger.info("rag_service.shutting_down")


app = FastAPI(
    title="糖尿病出院管理 RAG 知识问答服务",
    description="MySQL 高频问答快车道 + RAG 兜底：查询分类 → 检索策略选择 → "
                "Milvus 稠密/稀疏混合检索 → 父子块回溯 → 重排 → 生成；同时对外提供 REST 与 MCP Server。",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 需鉴权的路径前缀：REST 业务接口 + MCP 端点（/health、/docs 保持开放）
_PROTECTED_PREFIXES = ("/api/v1", "/rag")


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """内部服务鉴权：受保护路径必须携带正确的 X-API-Key，否则 401。"""
    if request.url.path.startswith(_PROTECTED_PREFIXES):
        provided = request.headers.get("X-API-Key", "")
        if not settings.rag_api_key:
            logger.error("rag_service.api_key_missing_config")
            return JSONResponse(
                {"detail": "服务未配置 RAG_API_KEY，拒绝访问"}, status_code=503
            )
        if not secrets.compare_digest(
            provided.encode("utf-8"), settings.rag_api_key.encode("utf-8")
        ):
            return JSONResponse({"detail": "无效或缺失的 API Key"}, status_code=401)
    return await call_next(request)

app.include_router(qa.router)
app.include_router(kb.router)

# MCP 端点挂载：FastMCP 内部端点为 /mcp，故完整端点为 http://host:8001/rag/mcp
app.mount("/rag", _kb_app)

# 前端页面（问答首页 / 知识库管理台）：用绝对路径定位 static，避免因启动目录不同而报错
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# 页面内 API Key 占位符：服务端渲染时替换为真实密钥，浏览器端无需手动输入。
# 说明：页面与本服务同源，密钥随页面下发；仅适用于内网 / 课程演示场景，
# 若对外公开部署，应改为真正的用户登录鉴权。
_KEY_PLACEHOLDER = "__RAG_API_KEY_JSON__"

# 禁止浏览器缓存页面：页面内联了 API Key 且随版本迭代，缓存会导致用户看到旧版页面
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _render_page(filename: str) -> HTMLResponse:
    """读取静态页面并把 API Key 注入占位符（避免把密钥硬编码进静态文件）。"""
    html = (_STATIC_DIR / filename).read_text(encoding="utf-8")
    html = html.replace(_KEY_PLACEHOLDER, json.dumps(settings.rag_api_key))
    return HTMLResponse(html, headers=_NO_CACHE_HEADERS)


@app.get("/", include_in_schema=False)
async def chat_page():
    """医学知识问答首页。"""
    return _render_page("index.html")


@app.get("/console", include_in_schema=False)
async def console_page():
    """知识库管理台（知识库重建 / 检索链路调试 / 评测说明）。"""
    return _render_page("console.html")


@app.get("/health", tags=["系统"])
async def health_check():
    return {"status": "ok", "service": "rag_service"}