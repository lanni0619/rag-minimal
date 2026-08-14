"""A + G — Augment：把檢索結果塞進 prompt；Generate：呼叫 LLM。"""

from typing import TypedDict
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

# LLM Global Behavior
SYSTEM_PROMPT = "你是維修助手，只能根據工具查到的文件回答問題，查到的文件若與問題不直接相關，優先修改問句重複調用工具重新查詢，而非反問使用者，若工具最終還是查不到適合的答案，可以根據工具查到的資料來提供使用者方向，但不要臆測或列出不相關資料供使用者篩選，沒有資料可參考時回覆『查無相關SOP資料，請向工程人員確認』。" \
    "\n\n## Clarification Boundary\n只有當問題完全沒有提到具體的機台名稱或故障現象（例如只說「幫我修一下」「壞了」），導致無法組成任何檢索關鍵字時，才直接反問使用者提供更多資訊，不要呼叫工具。\n只要問題包含任何具體的機台、部件、或現象描述，就視為足夠清楚，直接呼叫工具查詢，不要因為敘述簡短或未講明細節就反問。"

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
            content = ai_response.content[0]
            return content.get("text")
    except Exception as e:
        return f"[Generation error] - {e}"
