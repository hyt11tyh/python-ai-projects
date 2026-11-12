"""Streamlit 网页：上传资料、提问，并展示答案与出处片段。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_qa import config
from course_qa.pipeline import ask, index_stats, ingest, search


def _index_ready(index_dir: Path) -> bool:
    return (index_dir / "chunks.json").exists()


def _safe_stats(index_dir: Path) -> dict | None:
    if not _index_ready(index_dir):
        return None
    try:
        return index_stats(index_dir)
    except (FileNotFoundError, ValueError, OSError):
        return None


def _save_uploads(files) -> list[Path]:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for file in files:
        dest = config.UPLOAD_DIR / Path(file.name).name
        dest.write_bytes(file.getbuffer())
        saved.append(dest)
    return saved


def main() -> None:
    st.set_page_config(page_title="课程资料问答系统", page_icon="📚", layout="wide")
    st.title("课程资料问答系统")
    st.caption("本地 BM25 检索 + DeepSeek 生成，答案会标注到来源文件和片段编号。")

    index_dir = config.INDEX_DIR
    key_ok = config.has_api_key()

    with st.sidebar:
        st.subheader("准备资料")
        st.write("API 密钥：" + ("已配置" if key_ok else "未配置（仍可导入和检索，但不能生成答案）"))
        st.write("默认会导入 `data/sample` 中的示例讲义，也可换成你自己的课件 PDF。")

        uploaded = st.file_uploader(
            "上传 PDF / TXT / Markdown",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
        )
        use_sample = st.checkbox("同时包含示例讲义", value=not uploaded)
        chunk_size = st.slider("分块大小（字符）", min_value=200, max_value=1000, value=config.CHUNK_SIZE, step=50)
        chunk_overlap = st.slider("分块重叠（字符）", min_value=0, max_value=200, value=config.CHUNK_OVERLAP, step=10)

        if st.button("建立 / 更新索引", type="primary"):
            inputs: list[Path] = []
            if uploaded:
                inputs.extend(_save_uploads(uploaded))
            if use_sample or not inputs:
                inputs.append(config.SAMPLE_DIR)
            try:
                with st.spinner("正在读取并建立索引…"):
                    result = ingest(
                        inputs=inputs,
                        index_dir=index_dir,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                st.success(f"已导入 {result.documents} 份文档，共 {result.chunks} 个片段。")
            except Exception as exc:  # noqa: BLE001 - 展示给使用者
                st.error(f"导入失败：{exc}")

        stats = _safe_stats(index_dir)
        st.divider()
        st.subheader("当前索引")
        if not stats:
            st.info("还没有索引。点击上方按钮导入示例讲义即可开始。")
        else:
            st.write(f"文档 {stats['documents']} 份，片段 {stats['chunks']} 个。")
            for source, count in stats["sources"].items():
                st.write(f"- {source}（{count}）")

    if not _index_ready(index_dir):
        st.info("请先在左侧建立索引。如果只是试用，直接点「建立 / 更新索引」即可加载示例讲义。")
        return

    question = st.text_input("请输入问题", placeholder="例如：过拟合和欠拟合有什么区别？")
    top_k = st.slider("检索条数", min_value=1, max_value=10, value=config.TOP_K)
    col_ask, col_search = st.columns(2)
    do_ask = col_ask.button("生成回答", type="primary", disabled=not question.strip())
    do_search = col_search.button("只看检索结果", disabled=not question.strip())

    if do_ask:
        if not key_ok:
            st.warning("未配置 DEEPSEEK_API_KEY，已改为只展示检索结果。把密钥写入项目根目录的 .env 后即可生成回答。")
            do_search = True
        else:
            try:
                with st.spinner("正在检索并调用 DeepSeek…"):
                    result = ask(question, top_k=top_k, index_dir=index_dir)
                st.subheader("回答")
                st.markdown(result.answer)
                st.subheader("参考片段")
                _render_hits(result.hits)
            except Exception as exc:  # noqa: BLE001
                st.error(f"生成失败：{exc}")

    if do_search and not do_ask or (do_search and not key_ok):
        try:
            hits, backend = search(question, top_k=top_k, index_dir=index_dir)
            st.subheader(f"检索结果（{backend}）")
            _render_hits(hits)
        except Exception as exc:  # noqa: BLE001
            st.error(f"检索失败：{exc}")


def _render_hits(hits) -> None:
    if not hits:
        st.write("没有检索到相关片段。可以试着换一种问法，或检查资料是否已导入。")
        return
    for hit in hits:
        title = f"[{hit.rank}] {hit.chunk.chunk_id}　相关度 {hit.score:.2f}"
        with st.expander(title, expanded=hit.rank <= 2):
            st.markdown(hit.chunk.text)


# streamlit run course_qa/app.py 时 __name__ 为 __main__；
# 作为库被导入（例如冒烟测试）时不启动页面，避免副作用。
if __name__ == "__main__":
    main()
