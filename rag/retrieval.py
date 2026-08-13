"""R — Retrieve：把 query 轉向量、跟文件庫比對相似度。"""

import uuid
from langchain_qdrant import QdrantVectorStore
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


def retrieve(query: str, k: int = 2) -> list[tuple[str, str]]:
    """實際上是會去檢索向量資料庫"""

    scores: list[tuple[Document, float]] = vector_store.similarity_search_with_score(
        query, k
    )

    res = [
        (doc.metadata["id"], doc.page_content) for (doc, score) in scores if score > 0
    ]

    return res[:k]


def best_match_score(query: str) -> float:
    scores: list[tuple[Document, float]] = vector_store.similarity_search_with_score(
        query, 1
    )
    return scores[0][1] if scores else 0.0
