"""Redis 问答结果缓存。

命中缓存直接返回，降低重复问题对 LLM + 检索的开销；Redis 不可用时自动降级为直查。
"""
import json
from typing import Optional

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_client = None
_available = None


def _get_client():
    global _client, _available
    if _available is False:
        return None
    if _client is None:
        try:
            import redis
            s = get_settings()
            _client = redis.Redis(
                host=s.redis_host,
                port=s.redis_port,
                db=s.redis_db,
                password=s.redis_password or None,
                socket_connect_timeout=1,
                socket_timeout=1,
                decode_responses=True,
            )
            _client.ping()
            _available = True
        except Exception as e:
            logger.warning("cache.redis_unavailable | error=%s", e)
            _available = False
            _client = None
    return _client


def get_answer(question: str) -> Optional[str]:
    """查询缓存答案，未命中返回 None。"""
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get(f"rag:answer:{question}")
    except Exception:
        return None


def set_answer(question: str, answer: str) -> None:
    """写入缓存答案（失败静默降级）。"""
    client = _get_client()
    if client is None:
        return
    try:
        client.setex(f"rag:answer:{question}", get_settings().cache_ttl_seconds, answer)
    except Exception as e:
        logger.warning("cache.set_failed | error=%s", e)


def set_json(key: str, value, ttl: int = 0) -> None:
    """写入 JSON 值（用于缓存高频问答语料、分词结果等）。ttl<=0 表示不过期。"""
    client = _get_client()
    if client is None:
        return
    try:
        payload = json.dumps(value, ensure_ascii=False)
        if ttl and ttl > 0:
            client.setex(key, ttl, payload)
        else:
            client.set(key, payload)
    except Exception as e:
        logger.warning("cache.set_json_failed | key=%s | error=%s", key, e)


def get_json(key: str):
    """读取 JSON 值，未命中/不可用返回 None。"""
    client = _get_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None