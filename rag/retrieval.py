"""R — Retrieve：把 query 轉向量、跟文件庫比對相似度。"""

import math
from rag.embedder import embedder
from document import DOC_VECTORS

# def tokenize(text: str) -> list[str]:
#     """Chunking 字串"""
#     return [ch for ch in text if not ch.isspace() and ch not in "，、：。"]


def to_vector(text: str) -> list[float]:
    """ 使用 Embedder 將文字轉成向量表示 """
    return embedder.embed_query(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """計算兩個向量之間的相似程度"""
    size = len(a)

    norm_a = math.sqrt(sum([value * value for value in a]))  # a 向量長度
    norm_b = math.sqrt(sum([value * value for value in b]))  # b 向量長度

    dot = sum([a[i] * b[i] for i in range(size)])

    return dot / (norm_a * norm_b)


def retrieve(query: str, k: int = 2) -> list[tuple[str, str]]:
    """實際上是會去檢索向量資料庫"""

    query_vector = to_vector(query)

    scored = [
        (id, text, cosine_similarity(query_vector, doc_vector))
        for (id, text, doc_vector) in DOC_VECTORS
    ]

    scored.sort(key=lambda x: x[2], reverse=True)

    res = [(id, text) for (id, text, score) in scored if score > 0]

    return res[:k]


def best_match_score(query: str) -> float:
    query_vector = to_vector(query)
    scored = [cosine_similarity(query_vector, vector) for (id, text, vector) in DOC_VECTORS]
    return max(scored) if scored else 0.0