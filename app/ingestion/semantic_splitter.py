"""模型语义分块。

用达摩院 `nlp_bert_document-segmentation_chinese-base` 语义分割模型，把长文按
「语义边界」切成句子/段落，比纯字符切分更贴合语义，减少把一句话/一个要点切开。

与 splitter.py 的关系：splitter.py 的父子块默认用中文递归切分；若把
`settings.use_semantic_splitter` 打开，则父块先用本模块的语义分句结果拼装。
"""
import threading
from typing import List

from app.core.logger import get_logger

logger = get_logger(__name__)

_instance = None
_lock = threading.Lock()


def _load_pipeline(model_path: str):
    """加载文档语义分割 pipeline（modelscope）。"""
    from modelscope.pipelines import pipeline

    logger.info("semantic_splitter.loading | path=%s", model_path)
    return pipeline(task="document-segmentation", model=model_path, device="cpu")


def get_semantic_pipeline(model_path: str):
    """获取语义分割 pipeline 单例。"""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = _load_pipeline(model_path)
                logger.info("semantic_splitter.loaded")
    return _instance


def split_sentences(text: str, model_path: str) -> List[str]:
    """把长文按语义切成句子/段落列表；模型不可用时回退到按换行/句号切分。"""
    try:
        pipe = get_semantic_pipeline(model_path)
        result = pipe(text)
        segmented = result.get("text", "") if isinstance(result, dict) else str(result)
        sentences = [s.strip() for s in segmented.split("\n\t") if s.strip()]
        if sentences:
            return sentences
    except Exception as e:
        logger.warning("semantic_splitter.failed | error=%s", e)

    # 回退：按句末标点粗切
    import re
    parts = re.split(r"(?<=[。！？；])", text)
    return [p.strip() for p in parts if p.strip()]


def merge_to_chunks(sentences: List[str], chunk_size: int, overlap: int = 0) -> List[str]:
    """把句子列表按 chunk_size 组装为块，尽量不切断句子。"""
    chunks: List[str] = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) <= chunk_size:
            buf += sent
        else:
            if buf:
                chunks.append(buf)
            buf = sent if len(sent) <= chunk_size else sent[:chunk_size]
    if buf:
        chunks.append(buf)
    return chunks
