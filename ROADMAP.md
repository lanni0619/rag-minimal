# Agentic RAG 學習路線

從 `rag-minimal` 手刻的核心概念（信心門檻、LLM 自主重查）出發，逐步換裝成 `agentic-rag-for-dummies` 用的真實技術棧（Ollama + Qdrant + LangGraph + HuggingFace embeddings）。

## 階段一：真檢索地基

- [x] 1. 裝 `langchain_huggingface`，跑通 `HuggingFaceEmbeddings`，印出向量維度
- [x] 2. `retrieval.py` 的 `to_vector`/`cosine_similarity` 換成真 embedding 版本
- [x] 3. 裝 Qdrant，SOP 文件存進去，`retrieve()` 改呼叫 `similarity_search`

## 階段二：真決策工具

- [x] 4. `retrieve()` 包成 `@tool`，`bind_tools()` 讓 LLM 自己決定呼叫，取代手寫 `SEARCH_AGAIN:` 解析
- [x] 5. 裝 Ollama 本地模型，`generate()` 換成 `ChatOllama`（Gemini 保留對照）
- [x] 6. 加「問題不清楚就反問使用者」分支（對照 dummies README Step 6 Clarification Boundary）

## 階段三：真 orchestration 框架（工具熟悉，非深度整合）

- [x] 7. 讀懂 dummies README Step 7 `AgentState` 欄位（`tool_call_count`/`iteration_count` 對應自己寫的 `max_retries`）
- [x] 8. 把 `rag()` 迴圈改寫成最小 LangGraph `StateGraph`（單一 orchestrator node + tools node，跑通即可）
- [x] 9. 摸 `ToolNode`：用內建的 `ToolNode` 取代自己寫的 tool_map for-loop，看差在哪
- [x] 10. 摸 `Command (edgeless)` 條件路由：寫一個 `route_after_orchestrator_call` 風格的分支函式，體會「node 回傳決定下一步走哪」的模式
- [x] 11. 摸 checkpointer：`InMemorySaver()` 存一次 state，體會「中斷後從哪個 node 繼續」是怎麼運作的
- [ ] 12. 摸 subgraph 組合：`State`（外層）包 `AgentState`（內層）的巢狀 graph 是怎麼接起來的，不用真的用上

## 之後再看要不要繼續

hierarchical indexing、context compression、多輪對話記憶、multi-agent 平行 —— 這些留到真的遇到對應限制才做，現階段只是「知道有這個東西、知道它解決什麼」。

## 每個階段的引導
拆成小步驟 + 提示 + 示範程式碼，一次講一個步驟，讓我自己動手改