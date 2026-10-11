"""文档加载：把各类文档统一解析为纯文本（支持扫描版 PDF / 图片 OCR）。

具体解析在 ingestion/loaders.py（PDF / Word / PPT / 图片 / 文本）；
本模块只负责单文件与目录级的遍历，并统一返回 [{"content","source"}] 格式。
"""
import os
from typing import List

from app.core.logger import get_logger
from app.ingestion.loaders import load_file

logger = get_logger(__name__)


def load_document(path: str) -> List[dict]:
    """加载单个文档，返回 [{"content": str, "source": str}]。

    返回列表是为了兼容「单个文件可能产出多段」的约定；当前实现合并为一段。
    """
    source = os.path.basename(path)
    text = load_file(path)
    if not text or not text.strip():
        return []
    return [{"content": text, "source": source}]


def load_directory(dir_path: str) -> List[dict]:
    """递归加载目录下所有受支持文档。"""
    results: List[dict] = []
    for root, _, files in os.walk(dir_path):
        for fn in files:
            full = os.path.join(root, fn)
            try:
                results.extend(load_document(full))
            except Exception as e:
                logger.warning("loader.failed | file=%s | error=%s", fn, e)
    logger.info("loader.loaded_docs | total=%d", len(results))
    return results