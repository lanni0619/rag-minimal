"""RAG 逐步實作練習：串起 retrieval（R）+ generation（A/G）。"""

import time

from rag.generation import generate

if __name__ == "__main__":
    start_time = time.time()

    print("[LLM generating ...]\n\n")
    answer = generate("插入卡片時沒有反應，排查的SOP是?")
    print(answer + "\n\n")

    elapsed_time = time.time() - start_time
    print(f"花費{elapsed_time}秒")
