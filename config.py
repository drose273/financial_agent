# ============================================================
# config.py — 全局配置
# ============================================================
# 所有可配置项集中在这里，方便后续修改

import os

# Ollama 本地模型配置
# Ollama 默认运行在 http://localhost:11434
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

# 模型名称（请确保本机 ollama pull 了这个模型）
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2")

# Embedding 模型名称
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "llama3.2")

# ChromaDB 向量数据库配置
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME = "financial_knowledge"
RETRIEVAL_TOP_K = 3

# 对话记忆配置
MEMORY_WINDOW_SIZE = 10
SESSION_TIMEOUT_SECONDS = 3600

# 服务配置
HOST = "0.0.0.0"
PORT = 8000
