"""RAGAS 评估。

用 RAGAS 的四个核心指标评估 RAG 问答质量：
    faithfulness        忠实度（答案是否忠于检索上下文）
    answer_relevancy    答案相关性（答案是否切题）
    context_precision   上下文精确率（检索到的上下文是否相关）
    context_recall      上下文召回率（是否召回回答问题所需信息）

裁判模型：DeepSeek（OpenAI 兼容）；向量模型：本地 BGE-M3（与线上检索一致）。

评估数据格式（JSON 数组，每条含）：
    {"question": "...", "answer": "...", "contexts": ["..."], "ground_truth": "..."}

用法：
    cd rag_service
    python -m rag_assessment.rag_as --data ./rag_assessment/rag_evaluate_data.json
"""
import argparse
import json

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _build_llm():
    """裁判 LLM：DeepSeek（OpenAI 兼容）。"""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    s = get_settings()
    client = AsyncOpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)
    return llm_factory(s.llm_model, client=client, max_tokens=4096)


def _build_embeddings():
    """向量模型：本地 BGE-M3（与检索链路保持一致）。"""
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=get_settings().bge_m3_model_path)


def run_eval(data_path: str):
    """执行评估，返回 RAGAS 结果对象。"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    with open(data_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        raise ValueError("评估数据为空")

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
            "ground_truth": [r["ground_truth"] for r in rows],
        }
    )

    llm = _build_llm()
    emb = _build_embeddings()
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    logger.info("ragas.eval_start | samples=%d", len(rows))
    result = evaluate(dataset, metrics=metrics)
    logger.info("ragas.eval_done")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS 评估")
    parser.add_argument("--data", default="./rag_assessment/rag_evaluate_data.json")
    args = parser.parse_args()
    res = run_eval(args.data)
    print(res)
