"""RAG 逐步實作練習：串起 retrieval（R）+ generation（A/G）。"""

from rag.generation import build_prompt, generate, parse_search_again
from rag.retrieval import best_match_score, retrieve

# ========== Agentic Logics ==========


def retrieve_with_retry(query: str, best_score: float) -> list[tuple[str, str]]:
    if best_score < 0.4:  # 模糊區間，放寬 k 再查一次
        return retrieve(query, k=4)
    return retrieve(query)  # 分數夠高，正常查就好


# ================ RAG ================


def rag(query: str) -> str:
    max_retries = 3
    curr_query = query

    for i in range(max_retries):
        best_score = best_match_score(curr_query)
        
        retrieved_data = retrieve_with_retry(curr_query, best_score)
        
        augmented_data = build_prompt(curr_query, retrieved_data)
        
        answer = generate(augmented_data)

        new_query = parse_search_again(answer)

        print(
            f"第{i + 1}次迭代 問題={curr_query} 回應={answer}\n\n"
        )

        if not new_query:
            return answer


        curr_query = new_query

    return f"重新查詢{max_retries}次仍找不到相關資料"


if __name__ == "__main__":
    answer = rag("印表機無法列印")