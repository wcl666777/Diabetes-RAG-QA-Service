"""训练 BERT 查询分类器（通用知识 / 专业咨询）。

流程：
  bert-base-chinese（底座）→ 读取 JSONL → 8:2 划分 → Trainer 微调 → 保存到 models/bert_query_classifier

数据格式（每行一条 JSON）：{"query": "空腹血糖多少算正常", "label": 1}
  label：0=通用知识，1=专业咨询

用法（在 rag_service 目录下执行）：
    python scripts/train_classifier.py --data ./data/classifier_train.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from app.core.logger import get_logger

logger = get_logger(__name__)

RAG_ROOT = Path(__file__).resolve().parents[1]
BERT_BASE = RAG_ROOT / "models" / "bert-base-chinese"
OUTPUT_DIR = RAG_ROOT / "models" / "bert_query_classifier"
MAX_LEN = 128
LABEL_NAMES = ["通用知识", "专业咨询"]


class QueryDataset(Dataset):
    """把 tokenizer 编码 + 标签封装成 PyTorch 可识别的数据集。"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_data(path: str):
    """读取 JSONL，返回 (文本列表, 标签列表)。"""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["query"])
            labels.append(int(obj["label"]))
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(RAG_ROOT / "data" / "classifier_train.jsonl"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    texts, labels = load_data(args.data)
    logger.info(
        "classifier.data_loaded | total=%d | 通用=%d | 专业=%d",
        len(texts), labels.count(0), labels.count(1),
    )

    tokenizer = BertTokenizer.from_pretrained(str(BERT_BASE))
    model = BertForSequenceClassification.from_pretrained(str(BERT_BASE), num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("classifier.model_ready | device=%s", device)

    tr_t, va_t, tr_l, va_l = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    def encode(batch):
        return tokenizer(batch, truncation=True, padding=True, max_length=MAX_LEN)

    tr_enc, va_enc = encode(tr_t), encode(va_t)

    def compute_metrics(eval_pred):
        logits, lbls = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == lbls).mean())}

    # 版本自适应：transformers >= 4.46 用 eval_strategy，旧版本用 evaluation_strategy
    import inspect

    _ta_keys = inspect.signature(TrainingArguments.__init__).parameters
    _strategy_key = "eval_strategy" if "eval_strategy" in _ta_keys else "evaluation_strategy"

    training_args = TrainingArguments(
        output_dir=str(RAG_ROOT / "bert_results"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_steps=20,
        weight_decay=0.01,
        logging_dir=str(RAG_ROOT / "bert_logs"),
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        **{_strategy_key: "epoch"},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=QueryDataset(tr_enc, tr_l),
        eval_dataset=QueryDataset(va_enc, va_l),
        compute_metrics=compute_metrics,
    )

    logger.info("classifier.train_start")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info("classifier.saved | path=%s", OUTPUT_DIR)

    preds = np.argmax(trainer.predict(QueryDataset(va_enc, va_l)).predictions, axis=-1)
    print(classification_report(va_l, preds, target_names=LABEL_NAMES))
    print(confusion_matrix(va_l, preds))


if __name__ == "__main__":
    main()
