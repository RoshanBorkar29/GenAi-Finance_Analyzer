# src/vector_store.py
from pathlib import Path
import pickle
import faiss
import numpy as np

class VectorStore:
    def __init__(self, vector_dir="vector_store"):
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.vector_dir / "financial_report.index"
        self.chunk_path = self.vector_dir / "chunks.pkl"
        # Track the unique file hashes already processed
        self.registry_path = self.vector_dir / "processed_hashes.pkl"

    def is_duplicate(self, file_hash):
        """Checks if this document has already been indexed."""
        registry = self.load_registry()
        return file_hash in registry

    def load_registry(self):
        if self.registry_path.exists():
            with open(self.registry_path, "rb") as f:
                return pickle.load(f)
        return set()

    def save_to_registry(self, file_hash):
        registry = self.load_registry()
        registry.add(file_hash)
        with open(self.registry_path, "wb") as f:
            pickle.dump(registry, f)

    def create_or_append_index(self, embeddings):
        """Appends new vectors if index exists, otherwise creates a new one."""
        dimension = embeddings.shape[1]
        
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            index.add(embeddings)
        else:
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
        return index

    def save_index(self, index):
        faiss.write_index(index, str(self.index_path))

    def load_index(self):
        return faiss.read_index(str(self.index_path))

    def append_chunks(self, new_chunks):
        """Appends new text chunks to the existing list safely."""
        existing_chunks = []
        if self.chunk_path.exists():
            existing_chunks = self.load_chunks()
            
        combined_chunks = existing_chunks + new_chunks
        
        with open(self.chunk_path, "wb") as f:
            pickle.dump(combined_chunks, f)

    def load_chunks(self):
        with open(self.chunk_path, "rb") as f:
            return pickle.load(f)
        
    def search(self, index, query_embedding, k=5):
        distances, indices = index.search(query_embedding, k)
        return distances, indices