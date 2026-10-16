"""RAG 服务启动脚本。

在 rag_service 目录下执行（也可直接在 IDE 中运行本文件）：
    python run.py

等价于命令行：
    uvicorn app.main:app --host 0.0.0.0 --port 8001
"""
import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,  # 开发时可改为 True 以自动重载
    )


if __name__ == "__main__":
    main()
