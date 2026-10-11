"""结构化日志：JSON 格式，统一格式，便于追踪请求。

使用结构化 JSON 方便后续接入日志平台（如 Langfuse / ELK）。
"""
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_configed = False


class JsonFormatter(logging.Formatter):
    """把日志格式化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", log_dir: str = "./logs",
                      log_file: str = "rag_service.log") -> None:
    global _configed
    if _configed:
        return
    formatter = JsonFormatter()
    handlers: list[logging.Handler] = []

    # ① 控制台输出：本地/容器内实时查看
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    handlers.append(console)

    # ② 文件输出：落盘到项目 logs/ 目录，按大小轮转，避免单文件无限增长
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, log_file),
            maxBytes=10 * 1024 * 1024,   # 单文件 10MB
            backupCount=5,               # 最多保留 5 个历史文件
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    except Exception:
        pass  # 落盘失败不影响控制台日志

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers = handlers
    _configed = True


def get_logger(name: str = "rag_service") -> logging.Logger:
    return logging.getLogger(name)