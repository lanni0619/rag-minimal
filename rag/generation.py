"""A + G — Augment：把檢索結果塞進 prompt；Generate：呼叫 LLM。"""

import logging
import os
from dotenv import load_dotenv
from typing import Union

# gemini-3 回傳的 thinking part 沒有文字內容，SDK 的 response.text
# 快速存取器每次都會警告，這裡不需要那個提醒。
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def build_prompt(query: str, retrieved: list[tuple[str, str]]) -> str:
    """將答案與參考文件一起傳給AI，目的是避免AI產生幻覺"""

    context = (
        "\n\n".join(f"[{doc_id}] {text}" for doc_id, text in retrieved)
        or "(無相關文件)"
    )

    return (
        "你是維修助手，只能根據下方文件回答問題。\n\n"
        f"參考文件:\n{context}\n\n"
        "如果參考文件不足以回答要重新搜尋，只回覆這一行，不要加其他文字或標點：SEARCH_AGAIN: <你覺得更精準的搜尋關鍵字>\n\n"
        f"問題: {query}\n答案:"
    )


def parse_search_again(answer: str) -> Union[str, None]:
    is_again = answer.startswith("SEARCH_AGAIN:")

    if is_again:
        cut_len = len("SEARCH_AGAIN:")
        return answer[cut_len:].strip()

    return None


def generate(prompt: str) -> str:
    """G — call an LLM if one is reachable, else show the prompt it would receive."""
    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
        )
        return response.text or ""
    except Exception as e:
        print(str(e))
        return f"[離線模式，沒有接上 LLM，以下是原本要送給模型的 prompt]\n\n{prompt}"
