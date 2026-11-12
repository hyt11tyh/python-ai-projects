"""命令行入口：导入资料、检索、提问。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from course_qa import config
from course_qa.pipeline import ask, format_hits, index_stats, ingest, search


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="course-qa",
        description="课程资料问答系统：导入讲义，检索片段，并调用 DeepSeek 生成带出处的回答。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="读取 PDF/TXT/MD，分块并建立本地 BM25 索引")
    p_ingest.add_argument(
        "inputs",
        nargs="*",
        default=[str(config.SAMPLE_DIR)],
        help="文件或目录，默认使用 data/sample",
    )
    p_ingest.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    p_ingest.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    p_ingest.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)

    p_search = sub.add_parser("search", help="只检索相关片段，不调用大模型")
    p_search.add_argument("question")
    p_search.add_argument("--top-k", type=int, default=config.TOP_K)
    p_search.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)

    p_ask = sub.add_parser("ask", help="检索后调用 DeepSeek 生成带出处的回答")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=config.TOP_K)
    p_ask.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)

    p_stats = sub.add_parser("stats", help="查看当前索引中的文档和片段数量")
    p_stats.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)

    args = parser.parse_args(argv)

    try:
        if args.command == "ingest":
            result = ingest(
                inputs=args.inputs,
                index_dir=args.index_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            print(f"已导入 {result.documents} 份文档，切分为 {result.chunks} 个片段。")
            print(f"索引目录：{result.index_dir}")
            print("来源文件：")
            for source in result.sources:
                print(f"  - {source}")
            return

        if args.command == "search":
            hits, backend = search(args.question, top_k=args.top_k, index_dir=args.index_dir)
            print(f"检索后端：{backend}，命中 {len(hits)} 条。\n")
            print(format_hits(hits))
            return

        if args.command == "ask":
            result = ask(args.question, top_k=args.top_k, index_dir=args.index_dir)
            print(result.answer)
            print("\n—— 检索依据 ——")
            print(format_hits(result.hits))
            return

        if args.command == "stats":
            stats = index_stats(args.index_dir)
            print(f"索引目录：{stats['index_dir']}")
            print(f"检索后端：{stats['backend']}")
            print(f"文档数：{stats['documents']}，片段数：{stats['chunks']}")
            for source, count in stats["sources"].items():
                print(f"  - {source}: {count} 个片段")
            return
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
