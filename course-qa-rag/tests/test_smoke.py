"""不依赖外部 API 的冒烟测试：导入、分块、检索、模块可导入。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_qa.chunking import chunk_documents
from course_qa.documents import load_documents
from course_qa.pipeline import ingest, search
from course_qa.retriever import tokenize


def test_sample_documents_load_and_chunk(tmp_path: Path) -> None:
    docs = load_documents([ROOT / "data" / "sample"])
    assert len(docs) >= 5
    suffixes = {Path(doc.path).suffix.lower() for doc in docs}
    assert ".md" in suffixes
    assert ".txt" in suffixes
    assert ".pdf" in suffixes
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=80)
    assert len(chunks) >= len(docs)
    assert all(chunk.text.strip() for chunk in chunks)


def test_tokenize_handles_chinese() -> None:
    tokens = tokenize("什么是过拟合和欠拟合？")
    assert "过拟合" in tokens
    assert "欠拟合" in tokens
    assert "什么" not in tokens


def test_ingest_and_retrieve(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    result = ingest(inputs=[ROOT / "data" / "sample"], index_dir=index_dir)
    assert result.chunks > 0
    hits, backend = search("什么是过拟合？如何缓解？", top_k=5, index_dir=index_dir)
    assert backend == "bm25"
    assert hits
    blob = "\n".join(hit.chunk.text for hit in hits)
    assert "过拟合" in blob


def test_unknown_question_still_returns_ranked_hits(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    ingest(inputs=[ROOT / "data" / "sample"], index_dir=index_dir)
    hits, _ = search("Transformer 多头注意力的计算公式是什么？", top_k=3, index_dir=index_dir)
    # 允许检索到若干片段，生成阶段再拒绝作答；此处只要求调用成功
    assert isinstance(hits, list)


def test_app_module_importable() -> None:
    from course_qa import app as app_module
    from course_qa import cli, llm, pipeline

    assert hasattr(app_module, "main")
    assert hasattr(cli, "main")
    assert hasattr(pipeline, "ask")
    assert "只根据" in llm.SYSTEM_PROMPT or "只根据" in llm.SYSTEM_PROMPT.replace(" ", "")
    assert "无法回答" in llm.SYSTEM_PROMPT
