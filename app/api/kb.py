"""知识库管理接口：文档上传、文档导入、向量库统计。"""
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.core import milvus_store
from app.core.logger import get_logger
from app.ingestion.build import build_from_files, build_knowledge_base

router = APIRouter(prefix="/api/v1/kb", tags=["知识库管理"])
logger = get_logger(__name__)


class BuildRequest(BaseModel):
    directory: str = Field(..., description="文档目录绝对路径")


def _resolve_kb_dir(requested: str) -> str:
    """校验待构建目录必须位于白名单根目录内，返回规范化绝对路径。

    防止调用方传入任意路径（如 /etc）导致读取服务器上的任意文档。
    """
    root = os.path.realpath(get_settings().kb_allowed_dir)
    target = os.path.realpath(requested)
    if target != root and not target.startswith(root + os.sep):
        logger.warning("kb.build_dir_rejected | requested=%s | root=%s", requested, root)
        raise HTTPException(status_code=400, detail="directory 必须位于允许的知识库目录内")
    if not os.path.isdir(target):
        raise HTTPException(status_code=400, detail="directory 不存在或不是目录")
    return target


@router.post("/build")
async def build_kb(req: BuildRequest, background: BackgroundTasks):
    """异步触发知识库重建（大文档入库耗时，放后台执行）。"""
    # 同步执行（demo 数据量小）；生产可改为 background.add_task(build_knowledge_base, req.directory)
    directory = _resolve_kb_dir(req.directory)
    n = build_knowledge_base(directory)
    return {"status": "ok", "inserted_chunks": n}


@router.post("/upload")
async def upload_doc(request: Request, filename: str = Query(..., description="文件名（含扩展名）")):
    """上传单个文档并增量入库，body 为文件的原始字节流。

    只处理本次上传的文件（存入 data/uploads/ 后直接向量化入库），无需重建整个知识库；
    子块 id 由内容决定，重复上传同一文件会覆盖而非重复写入。多文件由前端逐个调用。

    说明：这里刻意不用 multipart（UploadFile），避免强依赖 python-multipart。
    """
    name = os.path.basename(filename.strip())
    if not name:
        raise HTTPException(status_code=400, detail="文件名无效")

    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")

    upload_dir = os.path.join(get_settings().kb_allowed_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, name)
    with open(dest, "wb") as out:
        out.write(data)

    # 解析 + 向量化属 CPU/IO 密集操作，放线程池避免阻塞事件循环
    inserted = await run_in_threadpool(build_from_files, [dest])
    logger.info("kb.upload_done | file=%s | bytes=%d | inserted=%d", name, len(data), inserted)
    return {
        "status": "ok",
        "file": name,
        "inserted_chunks": inserted,
        "doc_count": milvus_store.count_docs(),
    }


@router.get("/count")
async def kb_count():
    """查询当前向量库文档条数。"""
    return {"doc_count": milvus_store.count_docs()}