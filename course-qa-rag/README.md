# 课程资料问答系统（RAG）

一个面向实习简历的检索增强生成（RAG）项目：把课件、笔记或论文 PDF 切成片段，在本地用 BM25 检索，再调用 DeepSeek 根据检索结果回答，并标注出处。资料不足时会明确说“根据现有资料无法回答”，而不是编造。

仓库里已经放好一组“机器学习与信息检索基础”的示例讲义，克隆后就能跑通导入、检索和网页演示。你也可以把 `data/sample` 换成自己的课程 PDF，作为更贴近本人学习经历的作品。

## 为什么适合放进实习简历

投 AI / 算法 / 后端实习时，面试官更想看到一个**能跑、能讲清模块、有评测**的完整作品，而不是又一个没有外部资料约束的聊天机器人。本项目可以对应说明：

- **文档处理**：PDF / TXT / Markdown 解析，面向中文的分块（按标题、段落和句子切分，并保留重叠）。
- **本地检索**：默认使用 `jieba` + BM25（`rank-bm25`），在笔记本上即可离线运行，不依赖付费 Embedding API。
- **生成与约束**：对接 DeepSeek Chat API（`https://api.deepseek.com`，OpenAI 兼容协议），提示词要求“只根据检索上下文作答、引用文件名和片段编号、证据不足就承认不知道”。
- **应用层**：提供命令行（导入、检索、提问）和 Streamlit 网页（上传、问答、展示来源片段）。
- **评测意识**：内置约 20 道问题，脚本会比较“有检索”和“无检索”的回答，并先评估检索是否命中了正确讲义。

你可以在简历和面试中清楚说出自己负责的模块：数据导入与中文分块、BM25 检索、RAG 提示词与引用、评测对比。

## 环境要求

- Python 3.10 或更高版本（开发时使用 3.13）
- 一份 [DeepSeek API 密钥](https://platform.deepseek.com/)（只在生成答案时需要；导入和检索不需要）

可选：若要改用本地向量检索，再安装 `sentence-transformers`，并设置环境变量 `COURSE_QA_EMBEDDING_MODEL` 为本地模型名。不设置时一律走 BM25，**不会下载大型模型**。

## 安装

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置 API 密钥

复制示例环境文件，把密钥写进 `.env`（该文件已被 `.gitignore` 忽略，不要提交）：

```bash
cp .env.example .env
```

`.env` 中只需填写：

```
DEEPSEEK_API_KEY=你的密钥
```

也可以在终端里临时导出 `DEEPSEEK_API_KEY`。程序不会打印或记录密钥内容。

## 导入示例讲义并提问

```bash
# 读取 data/sample，建立本地索引（不调用大模型）
python -m course_qa ingest

# 只看检索结果，适合先检查有没有命中正确段落
python -m course_qa search "什么是过拟合？如何缓解？"

# 检索后调用 DeepSeek，生成带出处的回答
python -m course_qa ask "什么是过拟合？如何缓解？"

# 查看索引里有哪些文件、切了多少片段
python -m course_qa stats
```

导入其他目录或单份文件：

```bash
python -m course_qa ingest /path/to/your/notes
python -m course_qa ingest ./homework.pdf ./lecture.md
```

## 启动网页

```bash
source .venv/bin/activate
streamlit run course_qa/app.py
```

浏览器打开终端里提示的地址（默认是 http://localhost:8501）。可以在左侧上传自己的 PDF / TXT / MD，点击“建立 / 更新索引”，然后在中间提问。答案下方会展开显示命中的原文片段。如果还没有配置 API 密钥，网页仍可导入和检索，只是不能生成回答。

## 运行评测

```bash
python scripts/eval_compare.py
```

脚本会：

1. 重新导入 `data/sample`（可加 `--skip-ingest` 跳过）；
2. 对 `data/eval/questions.json` 里约 20 道题做检索命中评估；
3. 若已配置 `DEEPSEEK_API_KEY`，再对比“先检索再回答”和“不给资料直接问模型”的要点覆盖率，并检查资料外问题是否会拒绝作答；
4. 把报告写到 `results/eval_report.md` 和 `results/eval_report.json`。

没有密钥时也会生成报告，只是跳过第 3 步。这正好可以用来展示：评测并不只看生成文本漂不漂亮，还要看检索有没有找对段落。

## 示例资料说明

`data/sample` 中是一组自编的短讲义，主题集中在**机器学习与信息检索基础**，便于问答效果稳定：

| 文件 | 内容 |
| --- | --- |
| `01-机器学习概述.md` | 定义、和传统编程的区别、项目流程、数据划分 |
| `02-监督学习与无监督学习.md` | 分类 / 回归、聚类 / 降维、三类数据集的职责 |
| `03-线性模型与梯度下降.md` | 线性回归、逻辑回归、学习率、批量 / SGD / 小批量 |
| `04-决策树与过拟合.md` | 过拟合与欠拟合、剪枝、正则化、集成 |
| `05-神经网络基础.md` | 感知机、激活函数、ReLU、反向传播主线 |
| `06-模型评估指标.md` | 混淆矩阵、精确率 / 召回率 / F1、ROC 与 AUC |
| `07-特征工程基础.txt` | 标准化与归一化、缺失值、数据泄漏 |
| `08-信息检索与RAG简介.pdf` | BM25、向量检索、RAG 流程与幻觉 |

这些文本是为演示专门撰写的教学笔记，不是教材原文摘录。把它们替换成你自己有权使用的课件或实验报告后，项目会更有个人辨识度。请不要把不能外传的内部资料放到公开仓库里。

若需要重新生成 08 号 PDF，可安装 `reportlab` 后运行 `python scripts/make_sample_pdf.py`。

## 项目结构

```
course-qa-rag/
├── course_qa/                 # 核心 Python 包
│   ├── documents.py           # 加载 PDF / TXT / Markdown
│   ├── chunking.py            # 中文友好分块
│   ├── retriever.py           # 默认 BM25，可选本地向量检索
│   ├── llm.py                 # DeepSeek 调用与提示词
│   ├── pipeline.py            # 导入、检索、问答编排
│   ├── cli.py                 # 命令行入口
│   └── app.py                 # Streamlit 网页
├── data/sample/               # 示例讲义（可替换）
├── data/eval/questions.json   # 评测问题与期望要点
├── scripts/eval_compare.py    # 有检索 vs 无检索 对比
├── results/                   # 评测报告输出目录
├── indexes/                   # 本地索引（不入库）
├── tests/test_smoke.py        # 不依赖 API 的冒烟测试
├── requirements.txt
├── .env.example
└── README.md
```

## 可选的向量检索

默认检索后端是 BM25，推荐先用它把项目跑通。如果之后想演示语义检索，可以：

```bash
pip install sentence-transformers
export COURSE_QA_RETRIEVER=embedding
export COURSE_QA_EMBEDDING_MODEL=你本地已下载的模型名
python -m course_qa ingest
```

未安装或未设置模型时，程序会给出明确错误，而不会在后台悄悄下载大模型。

## 如何写进简历（可直接改写）

**课程资料问答系统（RAG）**｜Python / BM25 / DeepSeek / Streamlit

- 实现面向中文课件的导入与分块流水线，支持 PDF、TXT、Markdown，在本地用 BM25 检索相关片段，无需付费向量接口。
- 接入 DeepSeek Chat API，约束模型仅依据检索上下文作答并标注文件名与片段编号；资料不足时拒绝编造。
- 提供命令行与 Streamlit 页面，支持上传资料、提问并回看来源原文。
- 设计 20 道评测问题，对比有检索与无检索的要点覆盖，并单独评估检索命中率。

面试时建议准备三个细节：为什么默认用 BM25 而不是一上来上向量库、分块大小和重叠怎么选、以及你如何证明模型没有“看起来很流畅但其实在编造”。
