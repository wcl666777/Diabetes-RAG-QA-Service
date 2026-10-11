"""中文预处理。

统一分词口径：快车道 BM25 与知识库 BM25 都用这里的分词，保证检索一致性。
"""
from typing import List

import jieba


def preprocess_text(text: str) -> List[str]:
    """jieba 精确分词，返回词列表用于 BM25。"""
    return jieba.lcut(text)
