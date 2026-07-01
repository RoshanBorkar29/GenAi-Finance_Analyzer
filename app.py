import streamlit as st
from pathlib import Path
import hashlib

from src.parser import PDFParser
from src.image_understanding import ImageAnalyzer
from src.knowledge_base import KnowledgeBase
from src.chunking import ChunkProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag import FinancialRAG

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Financial Multimodal RAG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Multimodal RAG")
st.markdown("Analyze Financial Annual Reports using AI")

# ----------------------------
# Session State & Autoload Persistence Check
# ----------------------------
if "processed" not in st.session_state:
    # ✅ Checks if a previous index already exists physically on disk
    index_file = Path("vector_store/financial_report.index")
    chunk_file = Path("vector_store/chunks.pkl")
    
    if index_file.exists() and chunk_file.exists():
        st.session_state.processed = True
    else:
        st.session_state.processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("📂 Upload Annual Report")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    process_button = st.button("🚀 Process Document")

# ----------------------------
# Status Banner
# ----------------------------
st.subheader("System Status")

if st.session_state.processed:
    st.success("✅ Document Ready (Loaded from active vector store)")
else:
    st.info("Upload a Financial Report to begin.")

# ----------------------------
# Create Reports Folder
# ----------------------------
REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Process Document
# ----------------------------
if process_button:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        pdf_path = REPORT_DIR / uploaded_file.name

        # Read file bytes to calculate a unique MD5 hash fingerprint
        file_bytes = uploaded_file.getbuffer()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        # Instantiate VectorStore early to check duplicates
        vector_db = VectorStore()

        if vector_db.is_duplicate(file_hash):
            st.info("ℹ️ This document has already been processed and indexed. Skipping duplication!")
            st.session_state.processed = True
        else:
            with st.spinner("Processing New Annual Report..."):
                try:
                    # ---------------- Parser ----------------
                    parser = PDFParser()
                    parsed = parser.get_document(pdf_path)

                    # ---------------- Image Understanding ----------------
                    prompt = """
                    Analyze this financial image.
                    Identify:
                    - Chart type
                    - Financial values
                    - Trends
                    - Years
                    - Key insights
                    Return a concise description.
                    """
                    analyzer = ImageAnalyzer()
                    analyzer.analyze_folder(parsed["image_paths"], prompt)

                    # ---------------- Knowledge Base ----------------
                    kb = KnowledgeBase()
                    documents = kb.build_documents()

                    # ---------------- Chunking ----------------
                    chunker = ChunkProcessor()
                    chunks = chunker.split_documents(documents)

                    # ---------------- Embeddings ----------------
                    embedder = EmbeddingGenerator()
                    embeddings = embedder.create_embeddings(chunks)

                    # ---------------- Vector Store (Saves Safely) ----------------
                    # Append vectors safely to FAISS index instead of overwriting
                    index = vector_db.create_or_append_index(embeddings)
                    vector_db.save_index(index)
                    
                    # Append raw chunks text to storage list
                    vector_db.append_chunks(chunks)
                    
                    # Log this file fingerprint so it is never processed again
                    vector_db.save_to_registry(file_hash)

                    st.session_state.processed = True
                    st.success("✅ New document processed and added successfully!")

                except Exception as e:
                    st.error(f"❌ Error : {e}")

# ----------------------------
# Question Section
# ----------------------------
st.divider()
st.subheader("💬 Ask Questions")

question = st.text_input("Ask a question about the annual report")
ask_button = st.button("Ask")

if ask_button:
    if not st.session_state.processed:
        st.warning("Please upload and process a PDF first.")
    elif question.strip() == "":
        st.warning("Please enter a question!!")
    else:
        with st.spinner("Generating Answer..."):
            try:
                retriever = Retriever()
                rag = FinancialRAG()
                retrieved_docs = retriever.retrieve(question, k=5)
                answer = rag.generate_answer(question, retrieved_docs)
                
                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )
                
                st.subheader("Answer")
                st.write(answer)
                
                st.subheader("Sources")
                for i, doc in enumerate(retrieved_docs, start=1):
                    st.markdown(
                        f"""
                        **Source {i}**
                        - Type: {doc.metadata.get("type")}
                        - File: {doc.metadata.get("source")}
                        """
                    )
            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state.chat_history:
    st.divider()
    st.subheader("Chat history")
    for i, chat in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(f"Conversation {i}"):
            st.markdown("### ❓ Question")
            st.write(chat["question"])
            st.markdown("### 🤖 Answer")
            st.write(chat["answer"])