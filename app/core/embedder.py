"""BGE-M3 嵌入模块（dense + sparse 双输出）。

实现说明：
    用 `milvus_model.hybrid.BGEM3EmbeddingFunction` / FlagEmbedding 加载 BGE-M3 时，
    当前运行环境中 FlagEmbedding 与 transformers 4.51 存在 `dtype` 传参不兼容问题。
    为不破坏既有环境，这里直接用 transformers 的 AutoModel 手动复刻 BGE-M3 的双路输出：
      - dense ：取 [CLS] 隐状态并做 L2 归一化（1024 维，配合 Milvus 的 IP 度量等价余弦相似度）
      - sparse：relu(sparse_linear(hidden)) 得到每 token 权重，按 token id 取 max 聚合为 {token_id: weight}
    其中 sparse_linear.pt 取自 BGE-M3 权重包，与官方实现一致。

两者在 Milvus 内做混合检索（WeightedRanker 加权融合），取代「独立 BM25 语料」的做法。
"""
import os
import threading
from typing import Dict, List, Tuple

import numpy as np
import torch

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_instance = None
_lock = threading.Lock()

_MAX_LENGTH = 512   # 文档块仅数百字，512 足够且 CPU 下更快


class _M3Encoder:
    """BGE-M3 双路编码器（dense + sparse）。"""

    def __init__(self, path: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        self.model.eval()

        # sparse 权重层（bge-m3 官方 sparse_linear.pt，输出标量权重）
        hidden_size = self.model.config.hidden_size
        self.sparse_linear = torch.nn.Linear(hidden_size, 1, bias=True)
        state = torch.load(
            os.path.join(path, "sparse_linear.pt"), map_location="cpu", weights_only=True
        )
        self.sparse_linear.load_state_dict(state)
        self.sparse_linear.eval()

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.sparse_linear.to(self.device)
        self.special_ids = set(self.tokenizer.all_special_ids or [])

    @torch.no_grad()
    def encode(
        self, texts: List[str], batch_size: int = 12, max_length: int = _MAX_LENGTH
    ) -> Tuple[np.ndarray, List[Dict[int, float]]]:
        dense_out: List[np.ndarray] = []
        sparse_out: List[Dict[int, float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hidden = self.model(**enc).last_hidden_state  # [B, L, H]

            # ① dense：CLS + L2 归一化
            cls = hidden[:, 0]
            dense = torch.nn.functional.normalize(cls, p=2, dim=1)
            dense_out.extend(dense.cpu().float().numpy())

            # ② sparse：每 token 一个权重，按 token id 取 max 聚合
            token_weights = torch.relu(self.sparse_linear(hidden)).squeeze(-1)  # [B, L]
            input_ids = enc["input_ids"].cpu()
            attention = enc["attention_mask"].cpu()
            tw = token_weights.cpu()
            for b in range(len(batch)):
                ids, weights, mask = input_ids[b], tw[b], attention[b]
                row: Dict[int, float] = {}
                for t in range(ids.shape[0]):
                    if int(mask[t]) == 0:
                        continue
                    tid = int(ids[t])
                    if tid in self.special_ids:
                        continue
                    val = float(weights[t])
                    if val > 0:
                        row[tid] = max(row.get(tid, 0.0), val)
                sparse_out.append(row)

        return np.asarray(dense_out, dtype=np.float32), sparse_out


def _load_model() -> _M3Encoder:
    """真正的加载逻辑（同步，较慢，放线程池里跑）。"""
    path = get_settings().bge_m3_model_path
    logger.info("embedder.loading | path=%s", path)
    return _M3Encoder(path)


def get_embedder() -> _M3Encoder:
    """获取 BGE-M3 编码器单例。"""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = _load_model()
                logger.info("embedder.loaded")
    return _instance


def encode_documents(texts: List[str]) -> Tuple[np.ndarray, List[Dict[int, float]]]:
    """批量编码文档，返回 (dense 数组, sparse 字典列表)。"""
    return get_embedder().encode(texts)


def encode_query(query: str) -> Tuple[np.ndarray, Dict[int, float]]:
    """编码单条查询，返回 (dense 一维向量, sparse 字典)。"""
    dense, sparse_list = get_embedder().encode([query])
    return dense[0], sparse_list[0]
