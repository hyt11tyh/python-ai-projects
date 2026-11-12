"""从 PDF、TXT、Markdown 加载纯文本。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(frozen=True)
class Document:
    source: str
    path: str
    text: str


def iter_input_paths(inputs: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"找不到路径：{path}")
        if path.is_dir():
            files = [
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            ]
            if not files:
                raise FileNotFoundError(f"目录中没有 PDF/TXT/MD 文件：{path}")
            paths.extend(files)
        elif path.suffix.lower() in SUPPORTED_SUFFIXES:
            paths.append(path)
        else:
            raise ValueError(f"暂不支持的文件类型：{path.name}（仅支持 .pdf / .txt / .md）")
    # 去重但保持顺序
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def load_documents(inputs: list[str | Path]) -> list[Document]:
    documents: list[Document] = []
    for path in iter_input_paths(inputs):
        text = load_file(path)
        if not text.strip():
            continue
        documents.append(Document(source=path.name, path=str(path), text=text))
    if not documents:
        raise ValueError("没有从输入路径中读到可用文本，请检查文件是否为空或 PDF 是否为扫描件。")
    return documents


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"不支持的文件类型：{path}")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        cleaned = _normalize_text(extracted)
        if cleaned:
            pages.append(cleaned)
        else:
            pages.append(f"（第 {index} 页未提取到文本，可能是扫描版 PDF。）")
    return "\n\n".join(pages)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def _normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    lines = [line.strip() for line in text.splitlines()]
    compact: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank and compact:
                compact.append("")
            blank = True
            continue
        blank = False
        if compact and compact[-1] and _should_join(compact[-1], line):
            prev = compact[-1]
            joiner = "" if (_is_cjk(prev[-1]) or _is_cjk(line[0])) else " "
            compact[-1] = prev + joiner + line
        else:
            compact.append(line)
    return "\n".join(compact).strip()


def _line_units(text: str) -> float:
    return sum(1 if _is_cjk(ch) else 0.5 for ch in text)


def _should_join(previous: str, current: str) -> bool:
    """把 PDF 按行宽硬折的两行重新拼回去，避免把“若干”拆成两段。"""
    if re.search(r"[。！？；：.!?:]$", previous):
        return False
    if current.startswith(("#", "- ", "* ")) or re.match(r"^\d+\.", current):
        return False
    if previous.startswith(("#", "- ", "* ")):
        return False
    # 标题通常较短；只有上一行接近满行时，才视为被排版硬折开
    return _line_units(previous) >= 28
