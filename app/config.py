"""RAG 服务配置中心。

所有配置从 .env 读取（pydantic-settings 自动完成类型转换与校验）。
用法：from app.config import get_settings; settings = get_settings()
"""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """配置模型：每个字段对应 .env 中的一项（大小写不敏感）。"""

    # ── 服务监听 ──
    app_host: str = "0.0.0.0"
    app_port: int = 8001
    log_level: str = "INFO"
    log_dir: str = "./logs"             # 日志落盘目录（相对服务启动目录）

    # ── 大模型（DeepSeek，走 OpenAI 兼容接口）──
    deepseek_api_key: str = ""          # 必填：DeepSeek API Key
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"    # 对话模型名

    # ── Milvus（向量库）──
    milvus_host: str = "localhost"
    milvus_port: int = 19531
    collection_name: str = "medical_guidelines"
    embedding_dim: int = 1024           # BGE-M3 dense 向量维度

    # ── Redis（问答结果缓存）──
    redis_host: str = "localhost"
    redis_port: int = 6380
    redis_db: int = 0
    redis_password: str = ""
    cache_ttl_seconds: int = 3600       # 缓存过期时间（默认 1 小时）

    # ── 本地模型权重路径 ──
    # 说明：默认指向本地 models 目录，也可改为其他已下载的权重路径。
    bge_m3_model_path: str = "./models/bge-m3"
    reranker_model_path: str = "./models/bge-reranker-large"

    # ── 检索参数 ──
    hybrid_candidate_k: int = 20        # 稠密 + 稀疏 各自召回数
    rerank_top_k: int = 5               # 重排后返回条数
    recall_threshold: float = 0.35      # 重排后最高分低于该值视为低置信

    # ── MySQL（高频问答快车道）──
    db_host: str = "localhost"
    db_port: int = 3307
    db_name: str = "diabetes"
    db_user: str = "diabetes"
    db_password: str = "diabetes123"

    # ── 高频问答快车道参数 ──
    faq_table_name: str = "faq_qa"        # 高频问答表（学科/科室 + 问题 + 答案）
    faq_hit_threshold: float = 0.85       # BM25 softmax 命中阈值，超过则直答
    faq_enabled: bool = True              # 是否启用快车道

    # ── 父子块检索参数 ──
    parent_chunk_size: int = 1200         # 父块（大粒度，作为上下文）
    child_chunk_size: int = 300           # 子块（小粒度，用于检索命中）
    chunk_overlap: int = 50               # 切分重叠
    candidate_m: int = 2                  # 重排后返回的父块数（送入大模型）

    # ── 查询分类 / 检索策略 ──
    classifier_model_path: str = "./models/bert_query_classifier"
    enable_query_classifier: bool = True  # 是否启用 BERT 查询分类
    enable_strategy_selector: bool = True # 是否启用 LLM 检索策略选择
    service_phone: str = "12345"          # 兜底话术里的客服电话

    # ── 模型语义分块（默认关闭）──
    use_semantic_splitter: bool = False
    semantic_splitter_model_path: str = "./models/nlp_bert_document-segmentation_chinese-base"

    @property
    def database_url(self) -> str:
        """MySQL 连接串（本层用同步 PyMySQL）。"""
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """获取全局唯一配置对象。"""
    return Settings()