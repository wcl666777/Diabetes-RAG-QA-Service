"""会话历史持久化。

表：conversations(id, session_id, question, answer, timestamp)
- 每轮问答落库；
- 每次仅保留最近 N 轮（默认 5），超出自动删除，控制表体积；
- 提供读取 / 更新 / 清除，供多轮问答回放历史上下文。
"""
from typing import List, Optional

import pymysql

from app.core.logger import get_logger
from app.core.mysql_client import get_mysql_client

logger = get_logger(__name__)

_KEEP_ROUNDS = 5          # 保留最近 5 轮
_table_ready = False


def init_conversation_table() -> None:
    """建会话历史表（幂等）。MySQL 不可用则静默跳过。"""
    global _table_ready
    if _table_ready:
        return
    client = get_mysql_client()
    if client is None:
        return
    try:
        client.cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                INDEX idx_session_id (session_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        client.connection.commit()
        _table_ready = True
        logger.info("conversation.table_ready")
    except pymysql.MySQLError as e:
        logger.error("conversation.init_failed | error=%s", e)


def _fetch_recent(client, session_id: str, limit: int) -> List[dict]:
    """取最近 limit 轮，按时间正序返回。"""
    client.cursor.execute(
        "SELECT question, answer FROM conversations WHERE session_id = %s "
        "ORDER BY timestamp DESC LIMIT %s",
        (session_id, limit),
    )
    rows = [{"question": r[0], "answer": r[1]} for r in client.cursor.fetchall()]
    return rows[::-1]


def get_session_history(session_id: Optional[str], limit: int = _KEEP_ROUNDS) -> List[dict]:
    """读取会话历史（无 session_id 或不可用则返回空列表）。"""
    if not session_id:
        return []
    client = get_mysql_client()
    if client is None:
        return []
    try:
        return _fetch_recent(client, session_id, limit)
    except pymysql.MySQLError as e:
        logger.error("conversation.fetch_failed | error=%s", e)
        return []


def update_session_history(session_id: Optional[str], question: str, answer: str) -> None:
    """写入一轮问答，并裁剪至最近 N 轮。"""
    if not session_id:
        return
    client = get_mysql_client()
    if client is None:
        return
    try:
        client.cursor.execute(
            "INSERT INTO conversations (session_id, question, answer, timestamp) "
            "VALUES (%s, %s, %s, NOW())",
            (session_id, question, answer),
        )
        # 删除超出 N 轮的旧记录
        client.cursor.execute(
            "DELETE FROM conversations WHERE session_id = %s AND id NOT IN ("
            "  SELECT id FROM ("
            "    SELECT id FROM conversations WHERE session_id = %s "
            "    ORDER BY timestamp DESC LIMIT %s"
            "  ) AS sub)",
            (session_id, session_id, _KEEP_ROUNDS),
        )
        client.connection.commit()
    except pymysql.MySQLError as e:
        logger.error("conversation.update_failed | error=%s", e)
        client.connection.rollback()


def clear_session_history(session_id: str) -> bool:
    """清除指定会话的全部历史。"""
    client = get_mysql_client()
    if client is None:
        return False
    try:
        client.cursor.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
        client.connection.commit()
        return True
    except pymysql.MySQLError as e:
        logger.error("conversation.clear_failed | error=%s", e)
        client.connection.rollback()
        return False
