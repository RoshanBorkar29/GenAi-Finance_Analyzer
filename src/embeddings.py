# src/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Runs locally on your machine—no internet or cloud API required!
        self.model = SentenceTransformer(self.model_name)
    
    def get_texts(self, chunks):
        texts = []
        for chunk in chunks:
            texts.append(chunk.page_content)
        return texts

    def create_embeddings(self, chunks):
        texts = self.get_texts(chunks)
        
        # Generates matrix with 384 dimensions natively
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )
        embeddings = embeddings.astype(np.float32)

        return embeddings