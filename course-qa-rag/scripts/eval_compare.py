#!/usr/bin/env python3
"""对比“有检索”和“无检索”的回答，并评估检索是否命中相关讲义。

未配置 DEEPSEEK_API_KEY 时，仍会完成检索评估并写出报告，跳过模型生成部分。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_qa import config
from course_qa.llm import generate_without_retrieval
from course_qa.pipeline import ask, ingest, search

CST = timezone(timedelta(hours=8))


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions", data)
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"评测文件格式不正确：{path}")
    return questions


def cover_points(text: str, points: list[str]) -> tuple[list[str], list[str]]:
    haystack = text.lower()
    hit: list[str] = []
    miss: list[str] = []
    for point in points:
        needles = [part.strip().lower() for part in point.replace("／", "/").split("/") if part.strip()]
        if needles and any(needle in haystack for needle in needles):
            hit.append(point)
        else:
            miss.append(point)
    return hit, miss


def retrieval_ok(hits, item: dict) -> bool:
    sources = {hit.chunk.source for hit in hits}
    expected_sources = set(item.get("relevant_sources", []))
    keywords = item.get("retrieval_keywords", [])
    source_ok = True
    if expected_sources:
        source_ok = bool(sources & expected_sources)
    keyword_ok = True
    if keywords:
        blob = "\n".join(hit.chunk.text for hit in hits)
        keyword_ok = any(keyword in blob for keyword in keywords)
    if item.get("unanswerable"):
        # 不可解问题：不强制命中特定文件，只要系统稍后能拒绝编造即可
        return True
    return source_ok and keyword_ok


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# 检索增强 vs 无检索 对比报告",
        "",
        f"- 生成时间：{payload['generated_at']}（UTC+8）",
        f"- 评测问题数：{payload['n_questions']}",
        f"- 检索后端：{payload['backend']}",
        f"- 是否调用 DeepSeek：{'是' if payload['llm_ran'] else '否（未配置 API 密钥，仅完成检索评估）'}",
        "",
        "## 1. 检索评估",
        "",
        f"- 可回答问题的检索命中率：{payload['retrieval_hits']}/{payload['retrieval_total']}"
        f"（{payload['retrieval_rate']:.1%}）",
        "- 判定方式：检索前若干片段是否覆盖题目标注的来源文件或关键词。",
        "",
        "| 编号 | 问题 | 应命中来源 | 实际来源 | 检索是否命中 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['id']} | {row['question']} | {', '.join(row['relevant_sources']) or '（不强制）'} "
            f"| {', '.join(row['retrieved_sources']) or '无'} | {'是' if row['retrieval_ok'] else '否'} |"
        )

    lines.extend(["", "## 2. 生成对比", ""])
    if not payload["llm_ran"]:
        lines.extend(
            [
                "本次没有调用大模型。配置 `DEEPSEEK_API_KEY` 后重新运行：",
                "",
                "```bash",
                "python scripts/eval_compare.py",
                "```",
                "",
                "报告会补充：同一问题在“先检索再回答”和“不给资料直接问模型”两种设置下的要点覆盖率。",
                "对标注为“资料中没有答案”的问题，还会检查有检索时是否诚实地回答无法作答。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- 有检索时平均要点覆盖率：{payload['rag_coverage']:.1%}",
                f"- 无检索时平均要点覆盖率：{payload['no_rag_coverage']:.1%}",
                f"- 不可解问题中，有检索时正确拒绝作答：{payload['abstain_rag']}/{payload['abstain_total']}",
                f"- 不可解问题中，无检索时正确拒绝作答：{payload['abstain_no_rag']}/{payload['abstain_total']}",
                "",
            ]
        )
        for row in payload["rows"]:
            lines.extend(
                [
                    f"### {row['id']}. {row['question']}",
                    "",
                    f"- 类型：{'资料外问题' if row['unanswerable'] else '资料内问题'}",
                    f"- 期望要点：{'；'.join(row['expected_points'])}",
                    f"- 有检索覆盖要点：{row['rag_hit_n']}/{row['expected_n']}"
                    + (f"（未覆盖：{'；'.join(row['rag_miss'])}）" if row['rag_miss'] else ""),
                    f"- 无检索覆盖要点：{row['no_rag_hit_n']}/{row['expected_n']}"
                    + (f"（未覆盖：{'；'.join(row['no_rag_miss'])}）" if row['no_rag_miss'] else ""),
                    "",
                    "<details><summary>有检索的回答</summary>",
                    "",
                    row["rag_answer"] or "（无）",
                    "",
                    "</details>",
                    "",
                    "<details><summary>无检索的回答</summary>",
                    "",
                    row["no_rag_answer"] or "（无）",
                    "",
                    "</details>",
                    "",
                ]
            )

    lines.extend(
        [
            "## 3. 简要结论",
            "",
            payload["conclusion"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    (path.with_suffix(".json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_conclusion(payload: dict) -> str:
    if not payload["llm_ran"]:
        rate = payload["retrieval_rate"]
        return (
            f"在未调用大模型的情况下，可回答问题上的检索命中率为 {rate:.1%}。"
            "这说明示例讲义可以被 BM25 按问题找到对应段落。配置 DeepSeek API 密钥后重跑本脚本，"
            "即可对比有检索和无检索的生成质量，尤其关注资料外问题是否会被模型编造。"
        )
    rag = payload["rag_coverage"]
    base = payload["no_rag_coverage"]
    abstain = (
        f"对 {payload['abstain_total']} 道资料中没有答案的问题，"
        f"有检索时正确拒绝 {payload['abstain_rag']} 道，"
        f"无检索时正确拒绝 {payload['abstain_no_rag']} 道。"
    )
    if rag >= base:
        cmp = f"有检索的平均要点覆盖率（{rag:.1%}）不低于无检索（{base:.1%}）。"
    else:
        cmp = (
            f"本次有检索覆盖率（{rag:.1%}）低于无检索（{base:.1%}），"
            "需要检查分块是否切断了关键句，或提示词是否过度保守。"
        )
    return cmp + abstain


def looks_like_abstention(text: str) -> bool:
    clues = ["根据现有资料无法回答", "无法回答", "资料不足", "没有提到", "未提及", "不知道"]
    return any(clue in text for clue in clues)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估检索命中率，并对比有无 RAG 的生成结果。")
    parser.add_argument("--questions", type=Path, default=config.EVAL_PATH)
    parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--data-dir", type=Path, default=config.SAMPLE_DIR)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "eval_report.md")
    args = parser.parse_args()

    if not args.skip_ingest:
        ingest(inputs=[args.data_dir], index_dir=args.index_dir)

    questions = load_questions(args.questions)
    llm_ran = config.has_api_key()
    rows: list[dict] = []
    retrieval_hits = 0
    retrieval_total = 0
    rag_cover_sum = 0.0
    no_rag_cover_sum = 0.0
    graded = 0
    abstain_total = 0
    abstain_rag = 0
    abstain_no_rag = 0
    backend = "bm25"

    for item in questions:
        question = item["question"]
        hits, backend = search(question, top_k=args.top_k, index_dir=args.index_dir)
        ok = retrieval_ok(hits, item)
        if not item.get("unanswerable"):
            retrieval_total += 1
            retrieval_hits += int(ok)

        row = {
            "id": item.get("id"),
            "question": question,
            "unanswerable": bool(item.get("unanswerable")),
            "expected_points": item.get("expected_points", []),
            "expected_n": len(item.get("expected_points", [])),
            "relevant_sources": item.get("relevant_sources", []),
            "retrieved_sources": list(dict.fromkeys(hit.chunk.source for hit in hits)),
            "retrieval_ok": ok,
            "rag_answer": "",
            "no_rag_answer": "",
            "rag_hit_n": 0,
            "no_rag_hit_n": 0,
            "rag_miss": [],
            "no_rag_miss": [],
        }

        if llm_ran:
            rag_answer = ask(question, top_k=args.top_k, index_dir=args.index_dir).answer
            no_rag_answer = generate_without_retrieval(question)
            row["rag_answer"] = rag_answer
            row["no_rag_answer"] = no_rag_answer
            expected = item.get("expected_points", [])
            if item.get("unanswerable"):
                abstain_total += 1
                abstain_rag += int(looks_like_abstention(rag_answer))
                abstain_no_rag += int(looks_like_abstention(no_rag_answer))
                row["rag_hit_n"] = int(looks_like_abstention(rag_answer))
                row["no_rag_hit_n"] = int(looks_like_abstention(no_rag_answer))
                row["expected_n"] = 1
                row["rag_miss"] = [] if row["rag_hit_n"] else ["应拒绝作答"]
                row["no_rag_miss"] = [] if row["no_rag_hit_n"] else ["应拒绝作答"]
            elif expected:
                rag_hit, rag_miss = cover_points(rag_answer, expected)
                base_hit, base_miss = cover_points(no_rag_answer, expected)
                row["rag_hit_n"] = len(rag_hit)
                row["no_rag_hit_n"] = len(base_hit)
                row["rag_miss"] = rag_miss
                row["no_rag_miss"] = base_miss
                rag_cover_sum += len(rag_hit) / len(expected)
                no_rag_cover_sum += len(base_hit) / len(expected)
                graded += 1
        rows.append(row)
        print(f"[{item.get('id')}] 检索{'命中' if ok else '未命中'}：{question}")

    payload = {
        "generated_at": datetime.now(CST).strftime("%Y-%m-%d %H:%M"),
        "n_questions": len(questions),
        "backend": backend,
        "llm_ran": llm_ran,
        "retrieval_hits": retrieval_hits,
        "retrieval_total": retrieval_total,
        "retrieval_rate": (retrieval_hits / retrieval_total) if retrieval_total else 0.0,
        "rag_coverage": (rag_cover_sum / graded) if graded else 0.0,
        "no_rag_coverage": (no_rag_cover_sum / graded) if graded else 0.0,
        "abstain_total": abstain_total,
        "abstain_rag": abstain_rag,
        "abstain_no_rag": abstain_no_rag,
        "rows": rows,
        "conclusion": "",
    }
    payload["conclusion"] = build_conclusion(payload)
    write_report(args.output, payload)
    print(f"\n报告已写入：{args.output}")
    if not llm_ran:
        print("未检测到 DEEPSEEK_API_KEY，已跳过模型生成对比。")


if __name__ == "__main__":
    main()
