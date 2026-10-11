"""知识库管理接口：文档导入、向量库统计。"""
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from app.core import milvus_store
from app.core.logger import get_logger
from app.ingestion.build import build_knowledge_base

router = APIRouter(prefix="/api/v1/kb", tags=["知识库管理"])
logger = get_logger(__name__)


class BuildRequest(BaseModel):
    directory: str = Field(..., description="文档目录绝对路径")


@router.post("/build")
async def build_kb(req: BuildRequest, background: BackgroundTasks):
    """异步触发知识库重建（大文档入库耗时，放后台执行）。"""
    # 同步执行（demo 数据量小）；生产可改为 background.add_task(build_knowledge_base, req.directory)
    n = build_knowledge_base(req.directory)
    return {"status": "ok", "inserted_chunks": n}


@router.get("/count")
async def kb_count():
    """查询当前向量库文档条数。"""
    return {"doc_count": milvus_store.count_docs()}