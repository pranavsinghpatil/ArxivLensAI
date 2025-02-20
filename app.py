from sentence_transformers import SentenceTransformer
import streamlit as st
import pickle
from qa_system import generate_answer_huggingface
from vector_store import search_faiss, load_faiss_index
import subprocess
import os
import faiss
import hashlib
from main import process_pdf  # ✅ Import function to process PDF automatically
from utils import get_faiss_index_filename

# ✅ **Fix GRPC Error**
os.environ["GRPC_DNS_RESOLVER"] = "ares"

# ✅ **Load embedding model**
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token="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")

# ✅ **Define directories**
project_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(project_dir, "temp")
faiss_indexes_dir = os.path.join(project_dir, "faiss_indexes")
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(faiss_indexes_dir, exist_ok=True)

# ✅ **Streamlit UI setup**
st.title("📄 AI-Powered Research Assistant")
st.sidebar.header("Upload Your Research Paper")

# ✅ **PDF Upload Section**
uploaded_file = st.sidebar.file_uploader("Upload a Research Paper (PDF)", type="pdf")

if uploaded_file:
    # ✅ Save the uploaded PDF temporarily
    pdf_path = os.path.join(temp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("✅ PDF Uploaded Successfully!")

    # ✅ Generate unique FAISS filenames
    faiss_index_filename = os.path.join(faiss_indexes_dir, get_faiss_index_filename(pdf_path))
    chunks_filename = os.path.join(faiss_indexes_dir, f"chunks_{get_faiss_index_filename(pdf_path)}.pkl")

    # ✅ **Auto-create FAISS & Chunks if missing**
    if not os.path.exists(faiss_index_filename) or not os.path.exists(chunks_filename):
        st.warning("FAISS index or Chunks file missing. Automatically processing PDF...")
        process_pdf(pdf_path)  # ✅ Process PDF to generate FAISS index

    # ✅ **Load FAISS index & chunks**
    try:
        faiss_index, chunks = load_faiss_index(pdf_path)
        st.session_state.faiss_index = faiss_index
        st.session_state.chunks = chunks
        st.session_state.index_loaded = True
        st.success("✅ FAISS index loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"Error loading index: {str(e)}")
        st.stop()

# ✅ **Query Input**
query = st.text_input("🔎 Ask a question about the research paper:")

if st.button("Get Answer", key="get_answer_button") and query:
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
    elif not st.session_state.get("index_loaded", False):
        st.error("Please wait for the PDF to be processed.")
    else:
        # ✅ Search FAISS index
        retrieved_chunks = search_faiss(query, st.session_state.faiss_index, embedding_model, st.session_state.chunks)

        # ✅ Generate the final answer using Gemini or Hugging Face
        answer = generate_answer_huggingface(query, retrieved_chunks)

        st.subheader("🔍 Answer:")
        st.write(answer)
