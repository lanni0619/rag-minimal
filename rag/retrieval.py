"""R — Retrieve：把 query 轉向量、跟文件庫比對相似度。"""

import uuid
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from langchain_core.documents import Document

from rag.embedder import embedder
from document import DOCS

# ================= Vector Store =================
doc_ids, categorys, texts = zip(*DOCS)
metadatas = [{"id": id, "category": c} for (id, c) in list(zip(doc_ids, categorys))]

namespace = uuid.NAMESPACE_OID
vector_ids = [str(uuid.uuid5(namespace, id)) for id in list(doc_ids)]

vector_store = QdrantVectorStore.from_texts(
    texts=list(texts),
    embedding=embedder,
    metadatas=metadatas,
    collection_name="demo_collection",
    ids=vector_ids,
    url="http://localhost:6333",
)

@tool
def retrieve(query: str, k: int = 2) -> str:
    """
    1. 收到使用者的問題時第一步都先調用此函式來尋找相關資料。
    2. 辨識回傳結果是否「直接」對應到使用者的問題（分類、關鍵字不符即視為不足），而非模稜兩可的答案。
    3. 若不足以回答問題，禁止直接反問使用者補充描述；必須先修改問句關鍵字重新調用本函式，最多重試 2 次。
    4. 重試 2 次後仍無法直接回答，才回覆「查無相關資料」，不得用不相關文件拼湊答案。
    """

    scores: list[tuple[Document, float]] = vector_store.similarity_search_with_score(
        query, k
    )

    res = [
        f"[{doc.metadata['id']}] {doc.page_content}"
        for (doc, score) in scores
        if score > 0.75
    ]

    return "\n\n".join(res)