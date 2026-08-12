# Agentic RAG 學習路線

從 `rag-minimal` 手刻的核心概念（信心門檻、LLM 自主重查）出發，逐步換裝成 `agentic-rag-for-dummies` 用的真實技術棧（Ollama + Qdrant + LangGraph + HuggingFace embeddings）。

## 階段一：真檢索地基

- [x] 1. 裝 `langchain_huggingface`，跑通 `HuggingFaceEmbeddings`，印出向量維度
- [x] 2. `retrieval.py` 的 `to_vector`/`cosine_similarity` 換成真 embedding 版本
- [ ] 3. 裝 Qdrant，SOP 文件存進去，`retrieve()` 改呼叫 `similarity_search`

## 階段二：真決策工具

- [ ] 4. `retrieve()` 包成 `@tool`，`bind_tools()` 讓 LLM 自己決定呼叫，取代手寫 `SEARCH_AGAIN:` 解析
- [ ] 5. 裝 Ollama 本地模型，`generate()` 換成 `ChatOllama`（Gemini 保留對照）
- [ ] 6. 加「問題不清楚就反問使用者」分支（對照 dummies README Step 6 Clarification Boundary）

## 階段三：真 orchestration 框架

- [ ] 7. 讀懂 dummies README Step 7 `AgentState` 欄位（`tool_call_count`/`iteration_count` 對應自己寫的 `max_retries`）
- [ ] 8. 把 `rag()` 迴圈改寫成最小的 LangGraph `StateGraph`（單一 orchestrator node）

## 之後再看要不要繼續

hierarchical indexing、context compression、多輪對話記憶、multi-agent 平行。
