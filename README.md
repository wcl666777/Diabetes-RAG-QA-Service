# RAG 知识问答服务（子项目一）

糖尿病出院管理医学指南的 **RAG 问答系统**，对外提供 REST API 与 **MCP Server** 两种能力。

## 核心能力

- **文档解析**：PDF / Word / Markdown / TXT 统一提取文本（含扫描件 OCR）
- **父子块切分**：中文递归 / 语义分块，父块（大粒度）供上下文、子块（小粒度）供检索命中
- **混合检索**：BGE-M3 稠密 + 稀疏双路 → Milvus 融合 → bge-reranker 精排
- **意图识别**：自训练 BERT 分类器（通用知识 / 专业咨询）+ 领域关键词兜底
- **检索策略**：LLM 策略选择（直接检索 / HyDE / 子查询 / 回溯）
- **高频问答快车道**：MySQL 问答表 + BM25 打分 + Redis 缓存，命中直返标准答案
- **RAG 生成**：基于检索上下文 + 对话历史，DeepSeek 生成（支持 SSE 流式）
- **会话持久化**：多轮问答历史落 MySQL，仅保留最近 N 轮
- **MCP 暴露**：通过 FastMCP 把检索能力暴露给外部 Agent

## 问答三态

```
提问 → ① FAQ 快车道命中 → 直接返回标准答案（source=faq）
     → ② 未命中 → RAG（分类 → 策略 → 检索 → 重排 → 生成）
     → ③ 无结果 → 兜底话术
```

## 目录结构

```
rag_service/
├── app/
│   ├── main.py               # FastAPI 入口（REST + 模型预热）
│   ├── config.py             # pydantic-settings 配置中心
│   ├── core/
│   │   ├── embedder.py       # BGE-M3 编码（稠密 + 稀疏双路）
│   │   ├── reranker.py       # bge-reranker 精排
│   │   ├── milvus_store.py   # Milvus 封装
│   │   ├── retriever.py      # 混合检索主流程
│   │   ├── query_classifier.py   # BERT 意图识别
│   │   ├── strategy_selector.py  # LLM 检索策略选择
│   │   ├── faq_search.py     # 高频问答快车道
│   │   ├── conversation.py   # 会话历史持久化
│   │   ├── cache.py          # Redis 缓存
│   │   ├── prompts.py        # RAG prompt
│   │   ├── llm.py            # DeepSeek 生成
│   │   └── rag_system.py     # RAG 主流程编排
│   ├── ingestion/            # 加载 / 分块（含父子块、语义分块）/ 入库
│   ├── api/                  # qa.py（问答）/ kb.py（知识库管理）
│   └── mcp/server.py         # FastMCP Server
├── scripts/
│   ├── import_faq.py         # FAQ 问答对导入（CSV/TXT → faq_qa）
│   └── train_classifier.py   # 意图分类器训练
├── data/                     # 指南 PDF / FAQ CSV / 训练语料
└── models/                   # 本地模型权重
```

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env
python -m app.ingestion.build --dir ./data              # 导入知识库
python scripts/import_faq.py --file ./data/faq_qa.csv   # 导入 FAQ（可选）
uvicorn app.main:app --host 0.0.0.0 --port 8001         # REST + MCP 同进程（MCP 端点 /rag/mcp）
```

> MCP Server 已挂载在 REST 服务的 `/rag/mcp`（Agent 默认调用该地址），一般无需单独启动。
> `python -m app.mcp.server` 可在 8002 起一个独立 MCP 进程（可选）。
>
> 检索延迟调优：`HYBRID_CANDIDATE_K`（重排候选数）越大召回越全、CPU 重排越慢；默认 10。

## 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/rag/query | 非流式问答 |
| POST | /api/v1/rag/query/stream | SSE 流式问答 |
| POST | /api/v1/rag/session | 创建会话 ID |
| GET | /api/v1/rag/history/{session_id} | 读取会话历史 |
| DELETE | /api/v1/rag/history/{session_id} | 清除会话历史 |
| POST | /api/v1/kb/build | 知识库重建 |
| GET | /api/v1/kb/count | 向量库文档数 |
| MCP | /rag/mcp | `search_knowledge_base` 工具（Agent 调用） |
