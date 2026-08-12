from langchain_huggingface import HuggingFaceEmbeddings

model_name = "Qwen/Qwen3-Embedding-0.6B"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

embedder: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

if __name__ == "__main__":
    vector = embedder.embed_query("讀卡機故障")
    print(f"向量維度: {len(vector)}")
    print(f"前 5 個數字: {vector[:5]}")
