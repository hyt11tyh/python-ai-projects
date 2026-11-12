"""导入、索引与问答的编排层。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from course_qa import config
from course_qa.chunking import Chunk, chunk_documents
from course_qa.documents import load_documents
from course_qa.llm import generate_answer
from course_qa.retriever import Hit, build_retriever, load_index, save_index


@dataclass
class IngestResult:
    documents: int
    chunks: int
    index_dir: Path
    sources: list[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    answer: str
    hits: list[Hit]
    backend: str


def ingest(
    inputs: list[str | Path] | None = None,
    index_dir: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    backend: str | None = None,
) -> IngestResult:
    inputs = inputs or [config.SAMPLE_DIR]
    index_dir = Path(index_dir or config.INDEX_DIR)
    documents = load_documents(list(inputs))
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
    )
    backend = backend or config.RETRIEVER_BACKEND
    # 先验证检索器能构建，再落盘，避免写出坏索引
    build_retriever(chunks, backend=backend, embedding_model=config.EMBEDDING_MODEL)
    save_index(index_dir, chunks, backend=backend)
    sources = list(dict.fromkeys(doc.source for doc in documents))
    return IngestResult(
        documents=len(documents),
        chunks=len(chunks),
        index_dir=index_dir,
        sources=sources,
    )


def load_pipeline(index_dir: Path | None = None, backend: str | None = None):
    index_dir = Path(index_dir or config.INDEX_DIR)
    chunks, saved_backend = load_index(index_dir)
    chosen = backend or config.RETRIEVER_BACKEND or saved_backend
    retriever = build_retriever(
        chunks, backend=chosen, embedding_model=config.EMBEDDING_MODEL
    )
    return chunks, retriever


def search(
    question: str,
    top_k: int | None = None,
    index_dir: Path | None = None,
) -> tuple[list[Hit], str]:
    _chunks, retriever = load_pipeline(index_dir=index_dir)
    hits = retriever.search(question, top_k=top_k or config.TOP_K)
    return hits, getattr(retriever, "backend", "bm25")


def ask(
    question: str,
    top_k: int | None = None,
    index_dir: Path | None = None,
) -> Answer:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空。")
    hits, backend = search(question, top_k=top_k, index_dir=index_dir)
    answer = generate_answer(question, hits)
    return Answer(question=question, answer=answer, hits=hits, backend=backend)


def index_stats(index_dir: Path | None = None) -> dict:
    index_dir = Path(index_dir or config.INDEX_DIR)
    chunks, backend = load_index(index_dir)
    sources: dict[str, int] = {}
    for chunk in chunks:
        sources[chunk.source] = sources.get(chunk.source, 0) + 1
    return {
        "index_dir": str(index_dir),
        "backend": backend,
        "chunks": len(chunks),
        "documents": len(sources),
        "sources": sources,
    }


def format_hits(hits: list[Hit]) -> str:
    if not hits:
        return "（没有检索到相关片段。）"
    lines: list[str] = []
    for hit in hits:
        preview = hit.chunk.text.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:160] + "…"
        lines.append(
            f"{hit.rank}. {hit.chunk.chunk_id}  相关度 {hit.score:.2f}\n   {preview}"
        )
    return "\n".join(lines)
