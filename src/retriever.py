# src/retriever.py
import numpy as np
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore

class Retriever:
    # ✅ FIX: Changed to double underscores __init__ so variables are properly initialized
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()

    def encode_query(self, query):
        embedding = self.embedder.model.encode(
            query,
            convert_to_numpy=True
        )
        embedding = np.array([embedding], dtype=np.float32)
        return embedding

    def search(self, index, query_embedding, k=5):
        distances, indices = self.vector_store.search(
            index, query_embedding, k
        )
        return distances, indices

    def retrieve(self, query, k=5):
        index = self.vector_store.load_index()
        chunks = self.vector_store.load_chunks()
        query_embedding = self.encode_query(query)
        _, indices = self.search(index, query_embedding, k)
        
        retrieved_docs = []
        for idx in indices[0]:
            # Optional safety check to prevent IndexErrors if idx is out of bounds
            if 0 <= idx < len(chunks):
                retrieved_docs.append(chunks[idx])
                
        return retrieved_docs