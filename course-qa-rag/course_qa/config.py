"""项目路径与模型配置。不会把 API 密钥写入日志或打印到终端。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
UPLOAD_DIR = DATA_DIR / "uploads"
EVAL_PATH = DATA_DIR / "eval" / "questions.json"
INDEX_DIR = ROOT / "indexes" / "default"
RESULTS_DIR = ROOT / "results"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

CHUNK_SIZE = int(os.getenv("COURSE_QA_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("COURSE_QA_CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("COURSE_QA_TOP_K", "5"))

# 可选：设置为本地 sentence-transformers 模型名后，才启用向量检索。
# 默认留空，使用 BM25，无需下载大模型。
EMBEDDING_MODEL = os.getenv("COURSE_QA_EMBEDDING_MODEL", "").strip()
RETRIEVER_BACKEND = os.getenv("COURSE_QA_RETRIEVER", "bm25").strip().lower()


def has_api_key() -> bool:
    return bool(DEEPSEEK_API_KEY)


def require_api_key() -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "未检测到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入密钥，"
            "或在运行前导出环境变量。"
        )
    return DEEPSEEK_API_KEY
