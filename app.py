import streamlit as st
from pathlib import Path

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
# Session State
# ----------------------------

if "processed" not in st.session_state:
    st.session_state.processed = False

# ----------------------------
# Sidebar
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:

    st.header("📂 Upload Annual Report")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    process_button = st.button("🚀 Process Document")

# ----------------------------
# Status
# ----------------------------

st.subheader("System Status")

if st.session_state.processed:
    st.success("✅ Document Ready")
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

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing Annual Report..."):

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

                analyzer.analyze_folder(
                    parsed["image_paths"],
                    prompt
                )

                # ---------------- Knowledge Base ----------------

                kb = KnowledgeBase()

                documents = kb.build_documents()

                # ---------------- Chunking ----------------

                chunker = ChunkProcessor()

                chunks = chunker.split_documents(
                    documents
                )

                # ---------------- Embeddings ----------------

                embedder = EmbeddingGenerator()

                embeddings = embedder.create_embeddings(
                    chunks
                )

                # ---------------- Vector Store ----------------

                vector_db = VectorStore()

                index = vector_db.create_index(
                    embeddings
                )

                vector_db.save_index(index)

                vector_db.save_chunks(chunks)

                st.session_state.processed = True

                st.success("✅ Document processed successfully!")

            except Exception as e:

                st.error(f"❌ Error : {e}")

# ----------------------------
# Question Section
# ----------------------------

st.divider()

st.subheader("💬 Ask Questions")

question = st.text_input(
    "Ask a question about the annual report"
)

ask_button = st.button("Ask")

if ask_button:

    if not st.session_state.processed:

        st.warning("Please upload and process a PDF first.")
    elif question.strip() == "":
        st.warning("Please enter a question!!")
    else:
        with st.spinner("Generating Answer..."):

            try:

                retriever=Retriever()
                rag=FinancialRAG()
                retrieved_docs=retriever.retrieve(question,k=5)
                answer=rag.generate_answer(question,retrieved_docs)
                st.session_state.chat_history.append(
                    {
                        "question":question,
                        "answer":answer
                    }
                )
                st.subheader("Answer")
                st.write(answer)
                st.subheader("Sources")
                for i,doc in enumerate(retrieved_docs,start=1):
                    st.markdown(
                        f"""
                        **Source{i}**
                        -Type:{doc.metadata.get("type")}
                        -File:{doc.metadata.get("source")}
                        """
                    )
            except Exception as e:
                st.error(F"Error:{e}")

if st.session_state.chat_history:
    st.divider()
    st.subheader("Chat history")
    for i,chat in enumerate(reversed(st.session_state.chat_history),start=1):
        with st.expander(f"Conversation{i}"):
            st.markdown("### ❓ Question")

            st.write(chat["question"])

            st.markdown("### 🤖 Answer")

            st.write(chat["answer"])