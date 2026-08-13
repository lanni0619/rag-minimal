import os
from langchain_huggingface import HuggingFaceEmbeddings

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

model_name = "Qwen/Qwen3-Embedding-0.6B"
model_kwargs = {"device": "cpu", "token": HUGGING_FACE_TOKEN}
encode_kwargs = {"normalize_embeddings": False}

embedder: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)