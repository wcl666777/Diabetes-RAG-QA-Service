"""多格式文档加载。

支持格式：
    .txt/.md/.markdown   纯文本
    .docx                Word（正文段落 + 表格 + 内嵌图片 OCR）
    .pptx/.ppt           PPT（按位置排序的文本框/表格/图片）
    .pdf                 可提取文本直接用；扫描版大图走 OCR
    .jpg/.jpeg/.png/.bmp/.webp  图片直接 OCR

统一返回 [{"content": str, "source": str}]，与 ingestion.loader 的既有约定保持一致。
"""
import io
import os
from typing import List

from app.core.logger import get_logger

logger = get_logger(__name__)

# 页面内图片面积占比超过该阈值才触发整页 OCR（降低开销）
_PDF_OCR_THRESHOLD = (0.6, 0.6)

# 支持的扩展名
SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".docx", ".pptx", ".ppt", ".pdf",
                  ".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _decode_text(path: str) -> str:
    """文本文件解码：UTF-8 优先，回退 GB18030（兼容中文 Windows 导出）。"""
    with open(path, "rb") as f:
        data = f.read()
    for enc in ("utf-8", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _load_docx(path: str) -> str:
    """Word：正文段落 + 表格 + 内嵌图片 OCR。"""
    import docx
    from app.ingestion.ocr import ocr_image

    d = docx.Document(path)
    parts: List[str] = []
    for p in d.paragraphs:
        if p.text.strip():
            parts.append(p.text)

    # 段落内嵌图片 → OCR
    for rel in getattr(d.part, "rels", {}).values():
        target = getattr(rel, "target_part", None)
        if target is None:
            continue
        ctype = getattr(target, "content_type", "")
        if ctype.startswith("image/"):
            blob = getattr(target, "_blob", None) or getattr(target, "blob", None)
            if blob:
                txt = ocr_image(blob)
                if txt:
                    parts.append(txt)

    # 表格内容
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _load_pptx(path: str) -> str:
    """PPT：遍历幻灯片，按位置排序提取文本框/表格/图片文字。"""
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from app.ingestion.ocr import ocr_image

    prs = Presentation(path)
    parts: List[str] = []

    def _extract_shape(shape):
        # 组合形状递归
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for sub in shape.shapes:
                _extract_shape(sub)
            return
        # 图片 → OCR
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            try:
                txt = ocr_image(shape.image.blob)
                if txt:
                    parts.append(txt)
            except Exception:
                pass
            return
        # 文本框
        if shape.has_text_frame:
            text = shape.text_frame.text.strip()
            if text:
                parts.append(text)
        # 表格
        if getattr(shape, "has_table", False) and shape.has_table:
            for row in shape.table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

    for slide in prs.slides:
        # 按 (上, 左) 排序，尽量还原阅读顺序
        shapes = sorted(slide.shapes, key=lambda s: (s.top or 0, s.left or 0))
        for shape in shapes:
            _extract_shape(shape)
    return "\n".join(parts)


def _load_pdf(path: str) -> str:
    """PDF：优先提取文本；若某页文本稀少（扫描版），对该页位图执行 OCR。"""
    import fitz  # PyMuPDF
    from app.ingestion.ocr import ocr_image

    doc = fitz.open(path)
    parts: List[str] = []
    for page in doc:
        text = page.get_text().strip()
        # 页面图片面积占比，用于判断是否扫描版
        images = page.get_images(full=True)
        page_area = max(page.rect.width * page.rect.height, 1.0)
        img_area = 0.0
        for img in images:
            try:
                xref = img[0]
                rects = page.get_image_rects(xref)
                for r in rects:
                    img_area += r.width * r.height
            except Exception:
                continue
        ratio = img_area / page_area

        if text and ratio < _PDF_OCR_THRESHOLD[0]:
            parts.append(text)
        else:
            # 文本稀少或整页为大图 → 渲染成位图 OCR
            try:
                pix = page.get_pixmap(dpi=200)
                import numpy as np
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                ocr_text = ocr_image(arr)
                parts.append(ocr_text or text)
            except Exception as e:
                logger.warning("loader.pdf_ocr_failed | page=%s | error=%s", page.number, e)
                parts.append(text)
    doc.close()
    return "\n".join(p for p in parts if p)


def _load_image(path: str) -> str:
    """图片：直接 OCR。"""
    from app.ingestion.ocr import ocr_image
    return ocr_image(path)


# 扩展名 → 解析函数
_LOADERS = {
    ".txt": _decode_text,
    ".md": _decode_text,
    ".markdown": _decode_text,
    ".docx": _load_docx,
    ".pptx": _load_pptx,
    ".ppt": _load_pptx,
    ".pdf": _load_pdf,
    ".jpg": _load_image,
    ".jpeg": _load_image,
    ".png": _load_image,
    ".bmp": _load_image,
    ".webp": _load_image,
}


def load_file(path: str) -> str:
    """按扩展名分派到对应解析函数，返回纯文本（不支持的格式返回空串）。"""
    ext = os.path.splitext(path)[1].lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        logger.warning("loader.unsupported | ext=%s", ext)
        return ""
    return loader(path) or ""
