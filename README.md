# RAG 最小範例

零安裝看懂 RAG 的三步驟架構，跑起來再回頭看 `agentic-rag-for-dummies` 會輕鬆很多。進度對照見 [ROADMAP.md](ROADMAP.md)。

```bash
python main.py
```

## 現況：Prototype

手刻版 RAG，沒接框架、沒接 DB，只接了 LLM API。4 個檔案：

| 檔案 | 對應 RAG 步驟 | 在幹嘛 |
|---|---|---|
| `embedder.py` | - | 載入 `HuggingFaceEmbeddings`，提供 `embed_query()` |
| `retrieval.py` | **R**etrieve | 4 份假 SOP 文件轉向量存記憶體，`retrieve()` 用 cosine similarity 找最相關的 k 份 |
| `generation.py` | **A**ugment + **G**enerate | `build_prompt()` 塞入檢索結果；`generate()` 呼叫 Gemini（沒接上就印出 prompt） |
| `main.py` | 串接 | `rag()` 迴圈：分數太低就放寬 k 重查；模型回 `SEARCH_AGAIN:` 就換關鍵字重跑，最多 3 次 |

「重查」邏輯是手寫字串解析（`SEARCH_AGAIN:` + `parse_search_again()`），不是真的 Agent 工具呼叫。

### RAG 方塊圖
```
使用者問題 (query)
      │
      ▼
┌──────────────┐        ┌──────────────────────────┐
│  to_vector   │◀───────│  DOCS（4 份假 SOP 文件）  │
└──────┬───────┘        └──────────────────────────┘
       │ query 向量                  │
       ▼                             ▼ 每份文件也轉向量
       └────────────┬────────────────┘
                     ▼
           cosine_similarity（逐份文件比對）
                     │
                     ▼
        retrieve()：排序、取分數最高的 k 份
                     │
              [R] Retrieve 完成
                     ▼
        build_prompt()：把檢索到的文件塞進 prompt
                     │
              [A] Augment 完成
                     ▼
        generate()：呼叫 LLM（或離線印出 prompt）
                     │
              [G] Generate 完成
                     ▼
                  最終答案
```

## 之後：模組化與換裝真技術棧

依 [ROADMAP.md](ROADMAP.md) 逐步把上面的手刻版換掉：

- **Qdrant**：`DOC_VECTORS` 換成向量資料庫，`retrieve()` 改呼叫 `similarity_search`
- **`@tool` + `bind_tools()`**：`retrieve()` 包成工具讓 LLM 自己決定要不要呼叫，取代 `SEARCH_AGAIN:` 解析
- **Ollama**：`generate()` 換成 `ChatOllama`（Gemini 保留對照）
- **LangGraph**：`rag()` 迴圈改寫成最小的 `StateGraph`

檔案要不要拆資料夾，等這幾項真的落地、檔案數變多了再拆——現在 4 個檔案還不值得分。
