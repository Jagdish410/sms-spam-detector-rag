"""Embedding + FAISS retrieval of similar labeled messages (the 'R' in RAG)."""
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE = Path("models/embeddings.npy")


class Retriever:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.model = SentenceTransformer(MODEL_NAME)

        if CACHE.exists():
            emb = np.load(CACHE)
        else:
            print("Embedding all messages (one-time, ~1-3 min)...")
            emb = self.model.encode(
                self.df["text"].tolist(), batch_size=64,
                normalize_embeddings=True, show_progress_bar=True,
            )
            CACHE.parent.mkdir(exist_ok=True)
            np.save(CACHE, emb)

        emb = np.asarray(emb, dtype="float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])  # inner product == cosine (normalized)
        self.index.add(emb)

    def search(self, text: str, k: int = 5):
        q = self.model.encode([text], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [
            {
                "text": self.df.loc[i, "text"],
                "label": self.df.loc[i, "label"],
                "similarity": float(s),
            }
            for s, i in zip(scores[0], ids[0])
        ]
