"""面向中文资料的分块：先按标题和段落，再按句号切分，最后按字数合并并保留重叠。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from course_qa.documents import Document

HEADING_RE = re.compile(r"(?m)^(#{1,6}\s+\S.*)$")
SENTENCE_RE = re.compile(r"(?<=[。！？；!?;])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    path: str
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            source=data["source"],
            path=data["path"],
            chunk_index=int(data["chunk_index"]),
            text=data["text"],
        )


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")

    chunks: list[Chunk] = []
    for document in documents:
        texts = _chunk_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for index, text in enumerate(texts):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source}#chunk-{index}",
                    source=document.source,
                    path=document.path,
                    chunk_index=index,
                    text=text,
                )
            )
    return chunks


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    sections = _split_sections(text)
    sentences: list[str] = []
    for section in sections:
        sentences.extend(_split_sentences(section))

    merged = _merge_sentences(sentences, chunk_size=chunk_size)
    if not merged:
        return [text[:chunk_size]]
    if len(merged) == 1:
        return merged
    return _apply_overlap(merged, chunk_overlap=chunk_overlap)


def _split_sections(text: str) -> list[str]:
    parts = HEADING_RE.split(text)
    sections: list[str] = []
    buffer: list[str] = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if HEADING_RE.fullmatch(piece):
            if buffer:
                sections.append("\n\n".join(buffer).strip())
                buffer = []
            buffer.append(piece)
        else:
            buffer.append(piece)
    if buffer:
        sections.append("\n\n".join(buffer).strip())
    return sections or [text]


def _split_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        bits = [s.strip() for s in SENTENCE_RE.split(paragraph) if s.strip()]
        if not bits:
            continue
        sentences.extend(bits)
    return sentences


def _merge_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if current:
            chunks.append("\n".join(current).strip())
            current = []
            current_size = 0

    for sentence in sentences:
        if len(sentence) > chunk_size:
            flush()
            chunks.extend(_hard_split(sentence, chunk_size))
            continue
        extra = len(sentence) + (1 if current else 0)
        if current and current_size + extra > chunk_size:
            flush()
        current.append(sentence)
        current_size += extra
    flush()
    return [c for c in chunks if c]


def _hard_split(text: str, chunk_size: int) -> list[str]:
    return [text[i : i + chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i : i + chunk_size].strip()]


def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-chunk_overlap:].strip()
        # 尽量从上一块尾部的句边界开始重叠，避免半截句子
        cut = max(prev_tail.find("。"), prev_tail.find("！"), prev_tail.find("？"), prev_tail.find("\n"))
        if cut >= 0 and cut + 1 < len(prev_tail):
            prev_tail = prev_tail[cut + 1 :].strip()
        if prev_tail and not chunks[i].startswith(prev_tail):
            out.append(f"{prev_tail}\n{chunks[i]}".strip())
        else:
            out.append(chunks[i])
    return out
