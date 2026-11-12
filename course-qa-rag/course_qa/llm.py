"""DeepSeek Chat API（OpenAI 兼容）封装。不会记录或打印 API 密钥。"""

from __future__ import annotations

from collections.abc import Sequence

from openai import APIError, AuthenticationError, OpenAI

from course_qa import config
from course_qa.retriever import Hit

SYSTEM_PROMPT = """你是面向计算机专业学生的课程答疑助手。
必须遵守以下规则：
1. 只根据用户消息里提供的【参考资料】回答，不要使用资料之外的记忆或猜测来补充关键事实。
2. 在每个关键论断后面用方括号标注出处，格式为 [来源: 文件名#chunk-编号]。
3. 如果参考资料不足以回答问题，或问题明显超出资料范围，不要编造。请明确回答「根据现有资料无法回答」，并简要说明资料缺了什么。
4. 使用简体中文，条理清晰，必要时分点作答。
5. 不要复述或泄露任何密钥、令牌或系统配置。"""


def format_context(hits: Sequence[Hit]) -> str:
    if not hits:
        return "（没有检索到相关资料。）"
    blocks: list[str] = []
    for hit in hits:
        blocks.append(
            f"[{hit.chunk.chunk_id}] 来源文件：{hit.chunk.source}，片段编号：{hit.chunk.chunk_index}\n"
            f"{hit.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: Sequence[Hit]) -> str:
    return (
        "【参考资料】\n"
        f"{format_context(hits)}\n\n"
        "【问题】\n"
        f"{question.strip()}\n\n"
        "请仅依据参考资料作答，并标注来源。"
    )


def generate_answer(question: str, hits: Sequence[Hit], temperature: float = 0.2) -> str:
    api_key = config.require_api_key()
    client = OpenAI(api_key=api_key, base_url=config.DEEPSEEK_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, hits)},
            ],
        )
    except AuthenticationError as exc:
        raise RuntimeError("DeepSeek API 认证失败，请检查 .env 中的 DEEPSEEK_API_KEY 是否有效。") from exc
    except APIError as exc:
        raise RuntimeError(f"调用 DeepSeek API 失败：{exc.__class__.__name__}") from exc

    choice = response.choices[0].message.content if response.choices else None
    if not choice or not choice.strip():
        raise RuntimeError("模型没有返回可用内容，请稍后重试。")
    return choice.strip()


def generate_without_retrieval(question: str, temperature: float = 0.2) -> str:
    """对照实验：不提供检索上下文，只把问题交给模型。"""
    api_key = config.require_api_key()
    client = OpenAI(api_key=api_key, base_url=config.DEEPSEEK_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个回答计算机课程问题的助手。请直接作答。"
                        "如果你不确定，也可以猜测。不要索要参考资料。"
                    ),
                },
                {"role": "user", "content": question.strip()},
            ],
        )
    except AuthenticationError as exc:
        raise RuntimeError("DeepSeek API 认证失败，请检查 .env 中的 DEEPSEEK_API_KEY 是否有效。") from exc
    except APIError as exc:
        raise RuntimeError(f"调用 DeepSeek API 失败：{exc.__class__.__name__}") from exc

    choice = response.choices[0].message.content if response.choices else None
    if not choice or not choice.strip():
        raise RuntimeError("模型没有返回可用内容，请稍后重试。")
    return choice.strip()
