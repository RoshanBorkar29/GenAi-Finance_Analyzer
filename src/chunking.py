# src/chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkProcessor:

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # ✅ FIX: Automatically create and assign the splitter during initialization
        self.splitter = self.create_splitter()
        

    def create_splitter(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        return splitter
        
    def split_documents(self, documents):
        # Now self.splitter correctly points to the instantiated splitter object
        chunks = self.splitter.split_documents(documents)
        return chunks