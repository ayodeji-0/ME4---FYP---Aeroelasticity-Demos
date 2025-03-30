from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

def get_chunks_with_model(query, model_name, topic="cbt_flutter", top_k=3):
    model_path = f"chatbot/models/{model_name}"
    embedder = SentenceTransformer(model_path)

    base_path = f"chatbot/data/{topic}/{model_name}"
    with open(f"{base_path}/embedded_papers.json") as f:
        chunks = json.load(f)
    index = faiss.read_index(f"{base_path}/faiss.index")

    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k=top_k)
    return [chunks[str(i)] for i in I[0]]
