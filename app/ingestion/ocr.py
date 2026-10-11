"""OCR 引擎。

优先使用 rapidocr_paddle（GPU 加速），不可用时回退 rapidocr_onnxruntime（CPU）。
用于扫描版 PDF、图片、以及 Word / PPT 内嵌图片的文字识别。
"""
import threading
from typing import Union

import numpy as np

from app.core.logger import get_logger

logger = get_logger(__name__)

_instance = None
_lock = threading.Lock()


def get_ocr():
    """获取 OCR 引擎单例（首次初始化较慢，加锁避免重复初始化）。"""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                try:
                    from rapidocr_paddle import RapidOCR  # 优先 GPU 版
                    _instance = RapidOCR()
                    logger.info("ocr.engine | rapidocr_paddle")
                except ImportError:
                    from rapidocr_onnxruntime import RapidOCR  # CPU 回退
                    _instance = RapidOCR()
                    logger.info("ocr.engine | rapidocr_onnxruntime")
    return _instance


def ocr_image(img: Union[str, np.ndarray, bytes]) -> str:
    """对图片执行 OCR，返回识别文本（按行拼接）；失败或识别为空时返回空串。"""
    try:
        engine = get_ocr()
        result, _ = engine(img)
        if not result:
            return ""
        return "\n".join(str(line[1]) for line in result)
    except Exception as e:
        logger.warning("ocr.failed | error=%s", e)
        return ""
