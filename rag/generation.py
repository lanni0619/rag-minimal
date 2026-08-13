"""A + G — Augment：把檢索結果塞進 prompt；Generate：呼叫 LLM。"""

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage

from rag.retrieval import retrieve

load_dotenv()

# Initial llm model (gemini-3.1-flash-lite/gemini-3.5-flash-lite/gemma-4-31b-it)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, seed=42)

# Bind llm with tools
tools = [retrieve]
tool_map = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = "你是維修助手，只能根據 retrieve 工具查到的文件回答問題，不能用自己的知識回答。收到問題時一定要先呼叫 retrieve 查詢。"

MessagesType = list[SystemMessage | HumanMessage | AIMessage | ToolMessage]


def generate(prompt: str) -> str:
    """Call LLM to generate answer"""
    max_iterations = 5
    try:
        messages: MessagesType = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(prompt),
        ]

        for i in range(max_iterations):
            ai_response = llm_with_tools.invoke(messages)
            messages.append(ai_response)

            if not ai_response.tool_calls:
                break

            for tool_call in ai_response.tool_calls:
                selected_tool = tool_map[tool_call["name"]]
                tool_output = selected_tool.invoke(tool_call["args"])
                tool_message = ToolMessage(
                    content=tool_output, tool_call_id=tool_call["id"]
                )
                messages.append(tool_message)

        if isinstance(ai_response.content, str):
            return ai_response.content
        elif not ai_response.content:
            return "查無相關資料"
        else:
            content: dict[str, str] = ai_response.content[0]
            return content.get("text")
    except Exception as e:
        return f"[Generation error] - {e}"
