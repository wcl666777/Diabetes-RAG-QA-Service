"""DeepSeek 大模型调用封装（OpenAI 兼容接口）。

RAG 服务仅需「文本补全 / 流式补全」，直接用 openai 客户端足够轻量；
Agent 服务则使用 langchain 的 init_chat_model 以配合 LangGraph。
"""
from typing import Iterator, List, Optional

import tenacity

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_client = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        s = get_settings()
        _client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)
    return _client


def _messages(system: str, user: str, history: Optional[List[dict]] = None) -> List[dict]:
    msgs = [{"role": "system", "content": system}]
    if history:
        for h in history[-5:]:  # 最多保留 5 轮
            if h.get("question"):
                msgs.append({"role": "user", "content": h["question"]})
            if h.get("answer"):
                msgs.append({"role": "assistant", "content": h["answer"]})
    msgs.append({"role": "user", "content": user})
    return msgs


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=0.5, min=0.5, max=4),
    stop=tenacity.stop_after_attempt(3),
    reraise=True,
)
def generate(prompt: str, history: Optional[List[dict]] = None, system: str = "你是糖尿病随访管理助手。") -> str:
    """一次性生成，返回完整文本（失败自动指数退避重试）。"""
    s = get_settings()
    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=s.llm_model,
            messages=_messages(system, prompt, history),
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.error("llm.generate_failed | error=%s", e)
        raise


def generate_stream(prompt: str, history: Optional[List[dict]] = None, system: str = "你是糖尿病随访管理助手。") -> Iterator[str]:
    """流式生成，逐段返回文本。"""
    s = get_settings()
    client = get_client()
    stream = client.chat.completions.create(
        model=s.llm_model,
        messages=_messages(system, prompt, history),
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content if chunk.choices else None
        if content:
            yield content