from dotenv import load_dotenv
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from rag.retrieval import retrieve


# Environment Variable
load_dotenv()

# Initial llm model (gemini-3.1-flash-lite/gemini-3.5-flash-lite/gemma-4-31b-it)
tools = [retrieve]
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, seed=42)
llm_with_tools = llm.bind_tools(tools)

# LLM Global Behavior
SYSTEM_PROMPT = (
    "你是維修助手，只能根據工具查到的文件回答問題，查到的文件若與問題不直接相關，優先修改問句重複調用工具重新查詢，而非反問使用者，若工具最終還是查不到適合的答案，可以根據工具查到的資料來提供使用者方向，但不要臆測或列出不相關資料供使用者篩選，沒有資料可參考時回覆『查無相關SOP資料，請向工程人員確認』。"
    "\n\n## Clarification Boundary\n只有當問題完全沒有提到具體的機台名稱或故障現象（例如只說「幫我修一下」「壞了」），導致無法組成任何檢索關鍵字時，才直接反問使用者提供更多資訊，不要呼叫工具。\n只要問題包含任何具體的機台、部件、或現象描述，就視為足夠清楚，直接呼叫工具查詢，不要因為敘述簡短或未講明細節就反問。"
)

# ========== Build Langgrah Node ==========


def call_llm(state: MessagesState):
    """Orchestrator Node"""

    messages = state["messages"]
    ai_response = llm_with_tools.invoke(messages)

    return {"messages": [ai_response]}


def should_call_tools(state: MessagesState) -> Literal["call_tools", END]:
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tools"

    return END


# ========= Init Checkpointer & Agent ==========

checkpointer = InMemorySaver()
store = InMemoryStore()
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent_builder = StateGraph(MessagesState)
agent_builder.add_node(call_llm)
agent_builder.add_node("call_tools", ToolNode(tools))

agent_builder.add_edge(START, "call_llm")
agent_builder.add_conditional_edges("call_llm", should_call_tools, ["call_tools", END])
agent_builder.add_edge("call_tools", "call_llm")

agent = agent_builder.compile(
    checkpointer=checkpointer, store=store, interrupt_before=["call_tools"]
)


# ========= External Function ==========


def generate(prompt: str) -> str:
    """Call LLM to generate answer"""
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]

    try:
        agent.invoke({"messages": messages}, config)

        state_snapshot = agent.get_state(config)

        if not state_snapshot.next:
            response = state_snapshot.values

        while state_snapshot.next:
            response = agent.invoke(None, config)
            state_snapshot = agent.get_state(config)

        content: list | str = response["messages"][-1].content

        if isinstance(content, str):
            return content

        return content[0].get("text")

    except Exception as e:
        print(f"[LLM Generate error] {str(e)}")
        return f"[Generation error] - {e}"
