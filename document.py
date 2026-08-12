from rag.embedder import embedder

DOCS = [
    (
        "sop_ejector",
        "電商取貨機 E04 錯誤代碼：出貨機構卡件，通常是彈簧片鬆脫或包裹卡在滑軌，需重新校正彈出機構。",
    ),
    (
        "sop_paylane",
        "停車繳費機無法讀卡：通常是讀卡機積塵或磁條讀取頭故障，先用酒精棉清潔讀卡頭再測試。",
    ),
    (
        "sop_printer",
        "繳費機收據列印不出來：熱感應印表機缺紙或卡紙，需打開上蓋檢查紙捲並重新裝填。",
    ),
    (
        "sop_network",
        "設備斷網無法回傳交易紀錄：檢查4G模組訊號燈，重啟路由器，確認APN設定是否正確。",
    ),
]

DOC_VECTORS: list[tuple[str, str, list[float]]] = [
    (id, text, embedder.embed_query(text)) for (id, text) in DOCS
]
