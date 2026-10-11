"""MySQL 高频问答客户端。

定位：RAG 的「前置快车道」。先用 BM25 命中高频问答表中的问题，
命中则直接返回 SQL 里预置的标准答案（快、准、稳定）；
未命中才回退到 RAG 重新检索生成。

表结构：
    faq_qa(id, category, question, answer)
"""
from typing import List

import pymysql

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class MysqlClient:
    """MySQL 连接与高频问答表操作。"""

    def __init__(self) -> None:
        s = get_settings()
        self.table = s.faq_table_name
        try:
            self.connection = pymysql.connect(
                host=s.db_host,
                port=s.db_port,
                user=s.db_user,
                password=s.db_password,
                database=s.db_name,
                charset="utf8mb4",
                # 开启自动提交：避免长驻服务里一次 SELECT 后事务一直挂着，
                # 长期占用表元数据锁，导致后续 DDL / 大批量写入被阻塞。
                autocommit=True,
            )
            self.cursor = self.connection.cursor()
            logger.info("mysql.connected | %s:%s/%s", s.db_host, s.db_port, s.db_name)
        except pymysql.MySQLError as e:
            logger.error("mysql.connect_failed | error=%s", e)
            raise

    def create_table(self) -> None:
        """建高频问答表（幂等）。"""
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                category VARCHAR(64) NOT NULL COMMENT '科室/分类，用于过滤',
                question TEXT NOT NULL COMMENT '标准问题',
                answer TEXT NOT NULL COMMENT '标准答案',
                INDEX idx_category (category)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        self.connection.commit()
        logger.info("mysql.table_ready | table=%s", self.table)

    def fetch_questions(self) -> List[str]:
        """取出全部标准问题（用于构建快车道 BM25 语料）。"""
        self.cursor.execute(f"SELECT question FROM {self.table}")
        return [row[0] for row in self.cursor.fetchall()]

    def fetch_answer(self, question: str):
        """按标准问题精确取答案（BM25 命中最佳问题后调用）。"""
        self.cursor.execute(
            f"SELECT answer FROM {self.table} WHERE question = %s LIMIT 1", (question,)
        )
        row = self.cursor.fetchone()
        return row[0] if row else None

    def insert_faq(self, rows: List[dict]) -> int:
        """批量写入问答对（导入知识库 CSV 时调用），rows: [{category, question, answer}]。"""
        if not rows:
            return 0
        self.cursor.executemany(
            f"INSERT INTO {self.table} (category, question, answer) VALUES (%s, %s, %s)",
            [(r["category"], r["question"], r["answer"]) for r in rows],
        )
        self.connection.commit()
        logger.info("mysql.faq_inserted | rows=%d", len(rows))
        return len(rows)

    def close(self) -> None:
        try:
            self.cursor.close()
            self.connection.close()
        except Exception:
            pass


# ── 进程级单例（供快车道 / 会话历史复用同一连接）──
_client_instance = None
_client_failed = False


def get_mysql_client():
    """获取 MySQL 客户端单例；不可用时返回 None（上层据此降级，不阻断 RAG）。"""
    global _client_instance, _client_failed
    if _client_failed:
        return None
    if _client_instance is None:
        try:
            _client_instance = MysqlClient()
            _client_instance.create_table()
        except Exception as e:
            logger.warning("mysql.unavailable | error=%s", e)
            _client_failed = True
            return None
    return _client_instance
