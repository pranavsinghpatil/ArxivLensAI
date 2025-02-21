from sentence_transformers import SentenceTransformer
import streamlit as st
import pickle
from qa_system import generate_answer_huggingface
from vector_store import search_faiss, load_faiss_index
import subprocess
import os
import faiss
import hashlib
from main import process_pdf
from utils import get_faiss_index_filename, get_chunks_filename
import pandas as pd

os.environ["PYTHONUTF8"] = "1"
# ✅ Fix GRPC error
os.environ["GRPC_DNS_RESOLVER"] = "ares"

# ✅ Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token="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")

# ✅ Set up project directories
project_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(project_dir, "temp")
faiss_indexes_dir = os.path.join(project_dir, "faiss_indexes")
extracted_images_dir = os.path.join(project_dir, "extracted_images")
tables_dir = os.path.join(project_dir, "extracted_tables")

# ✅ Ensure directories exist
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(faiss_indexes_dir, exist_ok=True)
os.makedirs(extracted_images_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# ✅ Default Research Paper Path
default_paper_path = os.path.join(temp_dir, "Attention Is All You Need(default_research_paper).pdf")
# D:\Gits\re\temp\Attention Is All You Need(default_research_paper).pdf
# ✅ Streamlit UI setup
st.set_page_config(page_title="📄 AI-Powered Research Assistant", layout="wide")
st.title("🤖 AI-Powered Research Assistant")

# 📌 Sidebar - Research Paper Upload
st.sidebar.header("📄 Upload Your Research Papers")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# ✅ Track uploaded PDFs
if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = []

# ✅ Process uploaded PDFs
available_papers = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")

        # ✅ Ensure FAISS index is built
        try:
            faiss_index_filename = get_faiss_index_filename(pdf_path)
            faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)

            if not os.path.exists(faiss_index_path):
                st.info(f"🔄 Processing {uploaded_file.name} ...")
                process_pdf(pdf_path)
                st.success(f"✅ {uploaded_file.name} processed!")

            # ✅ Store available papers
            available_papers[uploaded_file.name] = pdf_path

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

# ✅ Load Default Paper if No Uploads
if not available_papers:
    st.sidebar.info("📌 Using Default Research Paper - \n \t Attention Is All You Need")
    if not os.path.exists(default_paper_path):
        st.error("⚠️ Default research paper is missing! Please upload a file.")
    else:
        available_papers["Attention Is All You Need (Default Paper)"] = default_paper_path

# ✅ Dropdown to Select Research Papers for Assistance
selected_papers = st.sidebar.multiselect(
    "📂 Select Research Papers to Assist On",
    list(available_papers.keys()),
    default=list(available_papers.keys())  # Select all by default
)

st.session_state.selected_papers = [available_papers[p] for p in selected_papers]

# ✅ Ensure Selected Papers are Processed
for pdf_path in st.session_state.selected_papers:
    try:
        faiss_index_filename = get_faiss_index_filename(pdf_path)
        faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)

        if not os.path.exists(faiss_index_path):
            process_pdf(pdf_path)

    except Exception as e:
        st.error(f"❌ Error processing {pdf_path}: {str(e)}")

# ✅ Query Input
query = st.text_input("🔎 Ask a question about the research papers:")

# ✅ Search and Generate Answer
if query:
    all_retrieved_chunks = []

    for pdf_path in st.session_state.selected_papers:
        try:
            faiss_index, chunks = load_faiss_index(pdf_path)
            retrieved_chunks = search_faiss(query, faiss_index, embedding_model, chunks)
            all_retrieved_chunks.extend(retrieved_chunks)
        except Exception as e:
            st.error(f"❌ Error retrieving from {pdf_path}: {str(e)}")

    # ✅ Generate the final answer using Gemini or Hugging Face
    answer = generate_answer_huggingface(query, all_retrieved_chunks)

    # ✅ Display the retrieved answer
    st.subheader("🔍 Answer:")
    st.write(answer)

# ✅ Extracted Tables & Images
for pdf_path in st.session_state.selected_papers:
    # 📊 Display Extracted Tables
    tables_file = os.path.join(tables_dir, f"tables_{get_faiss_index_filename(pdf_path)}.md")
    if os.path.exists(tables_file):
        st.subheader(f"📊 Extracted Tables from {os.path.basename(pdf_path)}")
        with open(tables_file, "r") as f:
            tables_content = f.read()
            st.markdown(f"```md\n{tables_content}\n```")

    # 🖼 Display Extracted Images
    image_files = [f for f in os.listdir(extracted_images_dir) if f.startswith(os.path.basename(pdf_path))]
    if image_files:
        st.subheader(f"🖼 Extracted Images from {os.path.basename(pdf_path)}")
        for image_file in image_files:
            image_path = os.path.join(extracted_images_dir, image_file)
            st.image(image_path, caption=f"Extracted Image: {image_file}", use_column_width=True)

# 📌 Display Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-message'>{chat['message']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-message'>{chat['message']}</div>", unsafe_allow_html=True)
