"""本地检索：默认 BM25（jieba 分词），可选本地向量模型。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from course_qa.chunking import Chunk

logger = logging.getLogger(__name__)

# 课程领域词，避免 jieba 把“过拟合”切成“过 / 拟合”
DOMAIN_TERMS = [
    "机器学习", "传统编程", "目标函数", "损失函数", "特征工程",
    "训练集", "验证集", "测试集", "交叉验证", "超参数", "泛化",
    "监督学习", "无监督学习", "半监督学习", "强化学习",
    "异常检测", "主成分分析", "线性回归", "逻辑回归", "均方误差",
    "交叉熵", "梯度下降", "学习率", "批量梯度下降", "随机梯度下降",
    "小批量梯度下降", "决策树", "信息增益", "基尼不纯度", "过拟合",
    "欠拟合", "预剪枝", "后剪枝", "正则化", "随机森林", "梯度提升",
    "感知机", "多层感知机", "激活函数", "反向传播", "梯度消失",
    "混淆矩阵", "真正例", "假正例", "真负例", "假负例", "准确率",
    "精确率", "召回率", "假正例率", "真正例率", "标准化", "归一化",
    "独热编码", "数据泄漏", "信息检索", "倒排索引", "逆文档频率",
    "向量检索", "余弦相似度", "检索增强生成", "幻觉",
]


def _init_jieba() -> None:
    jieba.setLogLevel(logging.WARNING)
    for term in DOMAIN_TERMS:
        jieba.add_word(term, freq=1000000)


_init_jieba()

STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "或",
    "及",
    "等",
    "把",
    "被",
    "就",
    "都",
    "而",
    "且",
    "也",
    "还",
    "很",
    "更",
    "最",
    "一个",
    "一种",
    "这个",
    "那个",
    "什么",
    "怎么",
    "如何",
    "为什么",
    "可以",
    "需要",
    "如果",
    "因为",
    "所以",
    "但是",
    "然后",
    "以及",
    "进行",
    "使用",
    "通过",
    "为了",
    "我们",
    "他们",
    "它",
    "其",
    "这",
    "那",
    "请",
    "简述",
    "介绍",
    "说明",
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "or",
    "in",
    "on",
    "for",
    "is",
    "are",
}


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(float(self.score), 4),
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "chunk_index": self.chunk.chunk_index,
            "text": self.chunk.text,
        }


class BM25Retriever:
    """基于 rank-bm25 的词法检索，完全离线，不调用任何付费 Embedding API。"""

    backend = "bm25"

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("没有可检索的片段，请先导入资料。")
        self.chunks = chunks
        self._tokenized = [tokenize(chunk.text) for chunk in chunks]
        # 避免空文档让 BM25 报错
        self._tokenized = [tokens or ["_empty_"] for tokens in self._tokenized]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        hits: list[Hit] = []
        for rank, (index, score) in enumerate(ranked[: max(1, top_k)], start=1):
            if score <= 0:
                continue
            hits.append(Hit(chunk=self.chunks[index], score=float(score), rank=rank))
        return hits


class EmbeddingRetriever:
    """可选的本地向量检索。需要自行安装 sentence-transformers 并指定模型名。"""

    backend = "embedding"

    def __init__(self, chunks: list[Chunk], model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise RuntimeError(
                "启用向量检索前请先安装可选依赖：pip install 'sentence-transformers'。"
                "若只是想先跑起来，请保持默认的 BM25（不要设置 COURSE_QA_EMBEDDING_MODEL）。"
            ) from exc

        self.chunks = chunks
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        texts = [chunk.text for chunk in chunks]
        self._embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        import numpy as np

        query_vec = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self._embeddings @ query_vec
        top_k = min(max(1, top_k), len(self.chunks))
        indices = np.argsort(-scores)[:top_k]
        hits: list[Hit] = []
        for rank, index in enumerate(indices, start=1):
            score = float(scores[int(index)])
            if score <= 0:
                continue
            hits.append(Hit(chunk=self.chunks[int(index)], score=score, rank=rank))
        return hits


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in jieba.cut(text):
        token = raw.strip().lower()
        if not token or token in STOPWORDS:
            continue
        if all(ch in " \t\n\r.,,，。！？、；：:（）()[]【】《》""''·—-_/\\|#+=*~`@%^&" for ch in token):
            continue
        tokens.append(token)
    return tokens


def build_retriever(
    chunks: list[Chunk],
    backend: str = "bm25",
    embedding_model: str = "",
):
    backend = (backend or "bm25").lower()
    if backend in {"embedding", "vector", "dense"}:
        if not embedding_model:
            raise RuntimeError(
                "已选择向量检索，但未设置 COURSE_QA_EMBEDDING_MODEL。"
                "请填写本地 sentence-transformers 模型名，或改回 bm25。"
            )
        logger.info("使用本地向量检索，模型：%s", embedding_model)
        return EmbeddingRetriever(chunks, model_name=embedding_model)
    return BM25Retriever(chunks)


def save_index(index_dir: Path, chunks: list[Chunk], backend: str) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": backend,
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    path = index_dir / "chunks.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_index(index_dir: Path) -> tuple[list[Chunk], str]:
    path = index_dir / "chunks.json"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到索引文件：{path}。请先运行 python -m course_qa ingest 导入资料。"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
    if not chunks:
        raise ValueError(f"索引为空：{path}")
    return chunks, payload.get("backend", "bm25")
