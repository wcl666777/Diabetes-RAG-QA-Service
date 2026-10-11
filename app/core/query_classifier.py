"""基于 BERT 的查询分类器。

作用：把用户查询分成两类，决定是否需要走「专业检索」：
- 通用知识：寒暄、常识类，直接由大模型回答，无需检索（省钱省时）；
- 专业咨询：涉及医学指南/用药/指标等，必须检索知识库后再生成（保证有据可依）。

模型：models/bert_query_classifier（二分类，num_labels=2），分词器复用
models/bert-base-chinese。
"""
import os
import threading

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_instance = None
_lock = threading.Lock()

# 标签映射：与训练时保持一致（通用知识 -> 0，专业咨询 -> 1）
LABEL_MAP = {"通用知识": 0, "专业咨询": 1}

# 医疗/糖尿病领域「强特征」关键词：命中即判「专业咨询」，保证一定走检索。
# 说明：bert_query_classifier 已在本项目语料上重新微调（data/classifier_train.jsonl），
# 分类边界交给模型判断，这里只保留一批「几乎只出现在医疗语境」的高精度词做兜底。
# 已剔除 饮食/运动/主食/空腹/餐后/睡前/监测/用药/剂量/医嘱/出院/复查/复诊/忌口/
# 注射/口服/血压/血脂 等日常也会用到或过于泛化的词，避免把闲聊误判为专业咨询。
_MEDICAL_KEYWORDS = [
    "糖尿病", "血糖", "胰岛素", "二甲双胍", "达格列净", "甘精", "糖化", "hba1c",
    "低血糖", "高血糖", "mmol", "酮症", "并发症", "视网膜", "肾病", "神经病变", "降糖",
]


class QueryClassifier:
    """BERT 查询分类器：加载 / 预测。"""

    def __init__(self, model_path: str = "") -> None:
        import torch
        from transformers import BertTokenizer

        s = get_settings()
        self.model_path = model_path or s.classifier_model_path
        # 分词器复用 bert-base-chinese（训练与推理保持一致）
        bert_base = os.path.join(os.path.dirname(self.model_path), "bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained(bert_base)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
        logger.info("query_classifier.ready | device=%s | path=%s", self.device, self.model_path)

    def _load_model(self) -> None:
        """优先加载已微调的模型；缺失则用 bert-base-chinese 初始化（未训练，效果有限）。"""
        from transformers import BertForSequenceClassification

        if os.path.exists(self.model_path) and os.path.exists(
            os.path.join(self.model_path, "config.json")
        ):
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("query_classifier.loaded | path=%s", self.model_path)
        else:
            base = os.path.join(os.path.dirname(self.model_path), "bert-base-chinese")
            self.model = BertForSequenceClassification.from_pretrained(base, num_labels=2)
            self.model.to(self.device)
            self.model.eval()
            logger.warning("query_classifier.init_new | path=%s（未微调，建议先训练）", base)

    def predict_category(self, query: str) -> str:
        """预测查询类别，返回 '通用知识' 或 '专业咨询'。

        安全兜底：模型不可用或推理异常时，返回 '专业咨询'——
        医疗场景下「漏检索」的代价远高于「多检索」，宁可多走一次 RAG。
        """
        import torch

        # 规则前置：命中医疗关键词直接判专业咨询（避免领域迁移误判导致漏检索）
        if any(kw in query for kw in _MEDICAL_KEYWORDS):
            return "专业咨询"

        if self.model is None:
            return "专业咨询"
        try:
            encoding = self.tokenizer(
                query, truncation=True, padding=True, max_length=128, return_tensors="pt"
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            with torch.no_grad():
                outputs = self.model(**encoding)
                prediction = torch.argmax(outputs.logits, dim=1).item()
            return "专业咨询" if prediction == 1 else "通用知识"
        except Exception as e:
            logger.warning("query_classifier.predict_failed | error=%s", e)
            return "专业咨询"


def get_classifier() -> QueryClassifier:
    """获取分类器单例（首次加载较慢，加锁避免并发重复加载）。"""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = QueryClassifier()
    return _instance
