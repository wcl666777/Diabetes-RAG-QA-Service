FROM python:3.11-slim

# 非交互安装 & 中文编码
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# 先装依赖（torch 体积较大，建议自带镜像层缓存或改本地构建缓存镜像）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝源码；模型权重通过 volume 挂载到 /app/models
COPY app ./app

EXPOSE 8001

# 默认启动 REST 服务；MCP Server 用 command 覆盖为 python -m app.mcp.server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]