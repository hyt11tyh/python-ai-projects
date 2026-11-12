#!/usr/bin/env python3
"""把 08 号讲义做成可检索的中文 PDF，演示 PDF 导入。

生成结果写入 data/sample/08-信息检索与RAG简介.pdf。
仅在需要重新生成示例 PDF 时运行，日常使用不必安装 reportlab。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scripts" / "assets" / "08-信息检索与RAG简介.md"
OUTPUT = ROOT / "data" / "sample" / "08-信息检索与RAG简介.pdf"
def _register_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    # STSong-Light 随 reportlab 提供，中英文混合文本也能被 pypdf 正确抽出。
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        pass

    fallback = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    if fallback.exists():
        pdfmetrics.registerFont(TTFont("CourseQAChinese", str(fallback)))
        return "CourseQAChinese"
    raise FileNotFoundError("未找到可用的中文字体，无法生成示例 PDF。")


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            current += char
            # 中文按字宽估算，ASCII 记一半
            units = sum(1 if ord(ch) > 127 else 0.5 for ch in current)
            if units >= width:
                lines.append(current)
                current = ""
        if current:
            lines.append(current)
    return lines


def main() -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise SystemExit("重新生成 PDF 需要：pip install reportlab") from exc

    if not SOURCE.exists():
        raise SystemExit(f"找不到源文件：{SOURCE}")

    font_name = _register_font()
    text = SOURCE.read_text(encoding="utf-8")
    output = OUTPUT
    width, height = A4
    left, top, leading = 64, height - 64, 18
    c = canvas.Canvas(str(output), pagesize=A4)
    c.setTitle("信息检索与 RAG 简介")
    c.setAuthor("课程资料问答系统示例讲义")
    c.setFont(font_name, 11)
    y = top
    for line in _wrap(text, width=36):
        if y < 64:
            c.showPage()
            c.setFont(font_name, 11)
            y = top
        if line.startswith("# "):
            c.setFont(font_name, 16)
            c.drawString(left, y, line[2:].strip())
            c.setFont(font_name, 11)
            y -= leading * 1.4
        elif line.startswith("## "):
            c.setFont(font_name, 13)
            c.drawString(left, y, line[3:].strip())
            c.setFont(font_name, 11)
            y -= leading * 1.2
        else:
            c.drawString(left, y, line)
            y -= leading
    c.save()
    print(f"已生成：{output}")


if __name__ == "__main__":
    main()
