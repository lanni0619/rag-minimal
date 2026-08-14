"""RAG 逐步實作練習：串起 retrieval（R）+ generation（A/G）。"""

from rag.generation import generate

TEST_CASES = [
    ("正常命中：問題與資料庫中的 SOP 直接對應", "列印收據時卡紙"),
    ("查無資料：現象具體，但資料庫中沒有對應 SOP", "使用者按下列印後，印表機沒有任何動作"),
    ("觸發反問：完全沒有機台或現象描述", "幫我修一下"),
    ("不觸發反問：有提到機台，即使沒講現象也直接查詢", "讀卡機好像有問題"),
    ("拒絕臆測：通用知識問題，不可用 LLM 自身知識回答", "晶片讀卡機跟磁條讀卡機哪個比較耐用?"),
    ("拒絕硬套：症狀與現有 SOP 方向相反，不可強行比附", "找零錢機一直吐錢"),
    ("複合查詢：一次問兩個類別，需分別查詢並整理回答", "收據卡紙零錢也不夠找"),
]

if __name__ == "__main__":
    for i, (label, question) in enumerate(TEST_CASES, start=1):
        print(f"{'=' * 60}")
        print(f"測項 {i}：{label}")
        print(f"問題：{question}")
        print(f"{'-' * 60}")
        print(generate(question))
        print()
