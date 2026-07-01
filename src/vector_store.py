# src/vector_store.py
from pathlib import Path
import pickle
import faiss
import numpy as np

class VectorStore:
    # ✅ FIX: Corrected to double underscores __init__
    def __init__(self, vector_dir="vector_store"):
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.vector_dir / "financial_report.index"
        self.chunk_path = self.vector_dir / "chunks.pkl"

    def create_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def save_index(self, index):
        faiss.write_index(index, str(self.index_path))

    def load_index(self):
        return faiss.read_index(str(self.index_path))

    def save_chunks(self, chunks):
        with open(self.chunk_path, "wb") as f:
            pickle.dump(chunks, f)

    def load_chunks(self):
        with open(self.chunk_path, "rb") as f:
            chunks = pickle.load(f)
        return chunks
        
    def search(self, index, query_embedding, k=5):
        distances, indices = index.search(query_embedding, k)
        return distances, indices