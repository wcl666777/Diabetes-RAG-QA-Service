# RAG 知识问答服务

> 面向糖尿病医学指南的**检索增强生成（RAG）问答系统**：把 PDF / Word 指南做成一个能理解中文医学问题、答案有据可依的问答服务，对外提供 **REST API** 与 **MCP Server** 两种能力。

> 本服务是「糖尿病出院患者智能随访助手」的**子项目一**，与子项目二（多 Agent 随访服务）通过 **MCP 协议**协作，为其提供医学知识检索能力。
> 姊妹项目：[Diabetes-Agent-Followup-Service](https://github.com/wcl666777/Diabetes-Agent-Followup-Service)

## 项目做了什么

一句话：**给医学问答做了一条「高频快车道 + 深度检索」的双层链路**。

- **快车道**：常见问题命中标准问答库（BM25 + 阈值）就直接返回预置答案——毫秒级、稳定可控；
- **深检索**：未命中才走完整 RAG —— 意图识别 → 检索策略选择 → BGE-M3 稠密+稀疏混合检索 → 父子块回溯 → 精排 → 生成。

这样既省掉大量不必要的模型调用，又保证专业问题**有据可依**、可追溯。

## 核心亮点

| 亮点 | 说明 |
|------|------|
| **父子块检索** | 子块（~300 字）参与向量/关键词命中，命中后回溯其父块（~1200 字）作为上下文，检索精度与上下文完整兼得 |
| **稠密 + 稀疏混合检索** | BGE-M3 双路向量 → Milvus `WeightedRanker(1.0, 0.7)` 融合 → bge-reranker 精排 |
| **BERT 意图识别** | 自训练 `bert-base-chinese` 分类器区分「通用知识 / 专业咨询」，仅专业咨询才检索，避免无谓开销 |
| **LLM 检索策略选择** | 大模型按问题特点在「直接检索 / HyDE / 子查询 / 回溯」中选一种；异常统一回退「直接检索」作安全网 |
| **高频问答快车道** | MySQL 问答表 + BM25 打分 + Redis 缓存，命中即直返，跳过整条 RAG 链路 |
| **BGE-M3 手动复刻** | 规避 FlagEmbedding 与 transformers 的 `dtype` 兼容问题，用官方 `sparse_linear` 权重手写稠密/稀疏双路编码 |
| **MCP 暴露** | FastMCP（stateless_http）把检索能力封装为 `search_knowledge_base` 工具，供上游 Agent 调用 |
| **健壮性** | Milvus / MySQL / Redis 任一不可用时逐级降级（缓存直查、快车道跳过），不阻断服务启动 |
| **可观测** | 结构化 JSON 日志，按大小轮转落盘 + 控制台双输出 |

## 技术栈

| 层面 | 选型 |
|------|------|
| Web / 服务 | FastAPI + Uvicorn + SSE-Starlette（流式） |
| 向量库 | Milvus 2.5（稠密+稀疏混合检索、WeightedRanker 融合、父块回溯去重） |
| 嵌入模型 | BGE-M3（transformers 手动复刻，稠密 1024 维 + 稀疏词权重） |
| 精排模型 | bge-reranker-large |
| 意图分类 | 自训练 BERT（bert-base-chinese 微调，HF Trainer） |
| 检索增强 | HyDE / 子查询 / 回溯 多策略（LLM 现场选择） |
| 结构化存储 | MySQL（FAQ 高频问答表 + 会话历史） |
| 缓存 | Redis |
| 大模型 | DeepSeek（OpenAI 兼容接口） |
| 对外协议 | REST + MCP（FastMCP，Streamable HTTP / JSON-RPC） |
| 文档解析 | pdfplumber / PyMuPDF / python-docx / python-pptx + RapidOCR（扫描件 OCR） |
| 评估 | RAGAS（faithfulness / answer_relevancy 等） |

## 检索链路

```
用户提问
  │
  ├─① 高频问答快车道（BM25 + 阈值）──命中──► 直接返回标准答案        （source=faq）
  │
  └─② 未命中 ──► BERT 意图识别
                    ├─ 通用知识 ──► 大模型直接回答（不检索）
                    └─ 专业咨询 ──► LLM 选检索策略
                                        └─► BGE-M3 稠密+稀疏 ──► Milvus 融合
                                              ──► 父块回溯去重 ──► bge-reranker 精排
                                              ──► DeepSeek 生成（REST 支持 SSE 流式）
  └─③ 无结果 ──► 兜底话术
```

## 目录结构

```
Diabetes-RAG-QA-Service/
├── app/
│   ├── main.py               # FastAPI 入口（REST + MCP 同进程挂载 + 模型预热）
│   ├── config.py             # pydantic-settings 配置中心
│   ├── core/
│   │   ├── embedder.py       # BGE-M3 编码（稠密 + 稀疏双路）
│   │   ├── reranker.py       # bge-reranker 精排
│   │   ├── milvus_store.py   # Milvus 封装（混合检索 / 去重 / 父块回溯）
│   │   ├── retriever.py      # 检索编排器（编码 → 混合检索 → 精排）
│   │   ├── query_classifier.py   # BERT 意图识别
│   │   ├── strategy_selector.py  # LLM 检索策略选择
│   │   ├── faq_search.py     # 高频问答快车道（BM25）
│   │   ├── conversation.py   # 会话历史持久化
│   │   ├── cache.py          # Redis 缓存
│   │   ├── prompts.py        # RAG prompt 模板
│   │   ├── llm.py            # DeepSeek 生成
│   │   └── rag_system.py     # RAG 主流程编排
│   ├── ingestion/            # 加载 / 分块（父子块、语义分块）/ 向量化入库
│   ├── api/                  # qa.py（问答）/ kb.py（知识库管理）
│   └── mcp/server.py         # FastMCP Server
├── scripts/
│   ├── import_faq.py         # FAQ 问答对导入（CSV/TXT → faq_qa）
│   └── train_classifier.py   # 意图分类器训练
├── rag_assessment/           # RAGAS 评估脚本
├── data/                     # 指南 PDF / FAQ CSV / 训练语料
└── models/                   # 本地模型权重（不随仓库提交）
```

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env
python -m app.ingestion.build --dir ./data              # 构建知识库（切分 + 向量化入库）
python scripts/import_faq.py --file ./data/faq_qa.csv   # 导入 FAQ（可选）
uvicorn app.main:app --host 0.0.0.0 --port 8001         # REST + MCP 同进程，MCP 端点 /rag/mcp
```

**前置依赖**：Milvus（向量库）、MySQL（快车道 + 会话）、Redis（缓存）、DeepSeek API Key；
本地模型权重（BGE-M3 / bge-reranker-large / bert_query_classifier）放到 `models/`。

> MCP Server 已挂载在 `/rag/mcp`（上游 Agent 默认调用该地址），一般无需单独启动；
> `python -m app.mcp.server` 可在 8002 起一个独立 MCP 进程（可选）。
>
> 检索延迟调优：`HYBRID_CANDIDATE_K`（重排候选数）越大召回越全、CPU 重排越慢，默认 10。

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

## 相关项目

- 姊妹项目：[Diabetes-Agent-Followup-Service](https://github.com/wcl666777/Diabetes-Agent-Followup-Service) —— 多 Agent 随访服务（本项目通过 MCP 为其提供检索能力）
- 总入口：[Diabetes-Discharge-Management-Assistant](https://github.com/wcl666777/Diabetes-Discharge-Management-Assistant) —— 含两个服务与 `docker-compose.yml`，可一键启动整套系统
