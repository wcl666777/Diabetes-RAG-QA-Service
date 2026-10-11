"""Milvus 向量库封装。

Schema 字段（父子块设计）：
    id              子块 ID（md5，幂等 upsert）
    content         子块正文（检索命中用）
    parent_id       所属父块 ID
    parent_content  所属父块完整内容（命中子块后回溯，作为上下文送大模型）
    source          来源文档
    dense_vector    BGE-M3 稠密向量（1024）
    sparse_vector   BGE-M3 稀疏向量（词权重）

检索：dense + sparse 两路 → WeightedRanker(1.0, 0.7) 加权融合 → 按 parent_content
去重 → 返回父块候选（供上层重排）。
"""
from typing import Dict, List, Optional

import numpy as np
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
    utility,
)

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_connected = False

# Milvus 输出字段
_OUTPUT_FIELDS = ["content", "parent_id", "parent_content", "source"]

# 混合检索权重：稠密（语义）1.0，稀疏（关键词）0.7
_DENSE_WEIGHT = 1.0
_SPARSE_WEIGHT = 0.7


def connect() -> None:
    """建立到 Milvus 的连接（幂等）。"""
    global _connected
    if _connected:
        return
    s = get_settings()
    connections.connect(alias="default", host=s.milvus_host, port=str(s.milvus_port))
    _connected = True
    logger.info("milvus.connected | host=%s:%s", s.milvus_host, s.milvus_port)


def _build_schema(dim: int) -> CollectionSchema:
    """构造父子块集合的 Schema。"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="parent_content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    return CollectionSchema(fields, description="医学指南知识库（父子块 + 稠密/稀疏向量）")


def _create_collection(name: str, dim: int) -> None:
    col = Collection(name, schema=_build_schema(dim))
    # 稠密：IVF_FLAT + 内积；稀疏：倒排索引 + 内积
    col.create_index(
        field_name="dense_vector",
        index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
    )
    col.create_index(
        field_name="sparse_vector",
        index_params={
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.2},
        },
    )
    col.load()
    logger.info("milvus.collection_created | name=%s", name)


def _needs_rebuild(name: str) -> bool:
    """检查已有集合是否为旧 schema（缺 sparse_vector / parent_content 字段）。"""
    try:
        col = Collection(name)
        field_names = {f.name for f in col.schema.fields}
        return not {"sparse_vector", "parent_content", "parent_id"}.issubset(field_names)
    except Exception:
        return True


def create_collection_if_not_exists() -> None:
    """建表（幂等）：若集合缺失或为旧 schema，则创建/重建。"""
    connect()
    s = get_settings()
    name = s.collection_name
    if utility.has_collection(name):
        if _needs_rebuild(name):
            logger.warning("milvus.schema_outdated | drop & rebuild | name=%s", name)
            utility.drop_collection(name)
        else:
            return
    _create_collection(name, s.embedding_dim)


def get_collection() -> Collection:
    create_collection_if_not_exists()
    col = Collection(get_settings().collection_name)
    col.load()
    return col


def insert_chunks(chunks: List[dict]) -> int:
    """批量插入子块。

    chunks: [{"id","content","parent_id","parent_content","source","dense","sparse"}]
    """
    if not chunks:
        return 0
    col = get_collection()
    data = [
        [c["id"] for c in chunks],
        [c["content"] for c in chunks],
        [c["parent_id"] for c in chunks],
        [c["parent_content"] for c in chunks],
        [c["source"] for c in chunks],
        [np.asarray(c["dense"], dtype=np.float32).tolist() for c in chunks],
        [c["sparse"] for c in chunks],
    ]
    col.insert(data)
    col.flush()
    logger.info("milvus.inserted | rows=%d", len(chunks))
    return len(chunks)


def _doc_from_hit(hit) -> dict:
    """把 Milvus 命中结果转成内部字典。"""
    e = hit.entity
    return {
        "content": e.get("content"),
        "parent_id": e.get("parent_id"),
        "parent_content": e.get("parent_content") or e.get("content"),
        "source": e.get("source"),
        "score": float(hit.distance),
    }


def _unique_parent_docs(hits: List[dict]) -> List[dict]:
    """按 parent_content 去重，保留父块内容作为上下文（命中子块 → 回溯父块）。"""
    seen = set()
    unique = []
    for h in hits:
        key = h["parent_content"]
        if key and key not in seen:
            seen.add(key)
            unique.append(
                {"content": key, "source": h.get("source", ""), "parent_id": h.get("parent_id", "")}
            )
    return unique


def hybrid_search(
    query_dense: np.ndarray,
    query_sparse: Dict[int, float],
    top_k: int = 20,
    source_filter: Optional[str] = None,
) -> List[dict]:
    """稠密 + 稀疏混合检索，返回去重后的父块候选 [{"content","source","parent_id"}]。"""
    col = get_collection()
    expr = f"source == '{source_filter}'" if source_filter else ""

    dense_req = AnnSearchRequest(
        data=[np.asarray(query_dense, dtype=np.float32).tolist()],
        anns_field="dense_vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        expr=expr,
    )
    sparse_req = AnnSearchRequest(
        data=[query_sparse],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {}},
        limit=top_k,
        expr=expr,
    )
    ranker = WeightedRanker(_DENSE_WEIGHT, _SPARSE_WEIGHT)

    results = col.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=ranker,
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
    )
    hits = []
    for hits_list in results:
        for hit in hits_list:
            hits.append(_doc_from_hit(hit))
    return _unique_parent_docs(hits)


def count_docs() -> int:
    connect()
    s = get_settings()
    if not utility.has_collection(s.collection_name):
        return 0
    return Collection(s.collection_name).num_entities
