# src/knowledge_base.py
from pathlib import Path
from langchain_core.documents import Document

class KnowledgeBase:

    # ✅ FIX: Corrected to double underscores __init__
    def __init__(self, markdown_dir="data/processed", image_desc_dir="data/extracted_text"):
        self.markdown_dir = Path(markdown_dir)
        self.image_desc_dir = Path(image_desc_dir)

    def load_markdown(self):
        documents = []
        markdown_files = self.markdown_dir.glob("*.md")
        for file in markdown_files:
            with open(file, "r", encoding="utf-8") as f:
                markdown = f.read()
            documents.append(
                Document(
                    page_content=markdown,
                    metadata={
                        "type": "markdown",
                        "source": file.name
                    }
                )
            )
        # ✅ FIX: Moved out of the loop
        return documents

    def load_image_descriptions(self):
        documents = []
        txt_files = self.image_desc_dir.glob("*.txt")
        for file in txt_files:
            with open(file, "r", encoding="utf-8") as f:
                description = f.read()
            documents.append(
                Document(
                    page_content=description,
                    metadata={
                        "type": "image",
                        "source": file.name
                    }
                )
            )
        # ✅ FIX: Added missing return statement
        return documents

    def build_documents(self):
        markdown_docs = self.load_markdown()
        image_docs = self.load_image_descriptions()
        documents = markdown_docs + image_docs
        return documents