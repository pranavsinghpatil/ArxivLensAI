#app.py
from sentence_transformers import SentenceTransformer
import streamlit as st
import pickle
from qa_system import generate_answer_huggingface
from vector_store import search_faiss, load_faiss_index
import os
import faiss
from main import process_pdf
from utils import get_faiss_index_filename, get_chunks_filename
import pandas as pd

# ✅ Fix GRPC error
os.environ["GRPC_DNS_RESOLVER"] = "ares"

# ✅ Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token="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")

# ✅ Set up directories
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

# ✅ Default Research Paper
default_paper_path = os.path.join(temp_dir, "Attention Is All You Need(default_research_paper).pdf")

# ✅ Streamlit UI setup
st.set_page_config(page_title="📄 AI-Powered Research Assistant", layout="wide")

# ✅ Display Fixed Header
st.markdown("<h3 style='text-align: center;'>🤖 AI-Powered Research Assistant</h3>", unsafe_allow_html=True)

# 📌 Sidebar - Multiple PDF Upload
st.sidebar.header("📄 Upload Research Papers")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# ✅ Store uploaded papers
available_papers = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ✅ Show notification instead of sidebar clutter
        st.toast(f"✅ {uploaded_file.name} uploaded!", icon="📄")

        # ✅ Process PDF if necessary
        try:
            faiss_index_filename = get_faiss_index_filename(pdf_path)
            faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)

            if not os.path.exists(faiss_index_path):
                process_pdf(pdf_path)

            available_papers[uploaded_file.name] = pdf_path
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

# ✅ Load Default Paper if No Uploads
if not available_papers:
    st.sidebar.info("📌 Using Default Research Paper - \n \t Attention Is All You Need")
    if os.path.exists(default_paper_path):
        available_papers["Attention Is All You Need (Default)"] = default_paper_path
    else:
        st.error("⚠️ Default research paper is missing! Please upload a file.")

# ✅ Dropdown to Select Research Papers
selected_papers = st.sidebar.multiselect(
    "📂 Select Research Papers",
    list(available_papers.keys()),
    default=list(available_papers.keys())
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

# ✅ Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Chatbot UI - Display Messages
chat_container = st.container()

# ✅ Display Chat History (Bottom to Top)
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user", avatar=os.path.join(project_dir, "icons", "user-icon.png")).markdown(message["content"])
    else:
        st.chat_message("assistant", avatar=os.path.join(project_dir, "icons", "bot-icon.png")).markdown(message["content"])

# ✅ Chat Input (Fixed at Bottom)
query = st.chat_input("Ask your question here...")

# ✅ Search and Generate Answer
if query:
    # ✅ Display user message in chat message container
    st.chat_message("user", avatar=os.path.join(project_dir, "icons", "user-icon.png")).markdown(query)
    # ✅ Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    all_retrieved_chunks = []
    for pdf_path in st.session_state.selected_papers:
        try:
            faiss_index, chunks = load_faiss_index(pdf_path)
            retrieved_chunks = search_faiss(query, faiss_index, embedding_model, chunks)
            all_retrieved_chunks.extend(retrieved_chunks)
        except Exception as e:
            st.error(f"❌ Error retrieving from {pdf_path}: {str(e)}")

    # ✅ Generate the final answer
    answer = generate_answer_huggingface(query, all_retrieved_chunks)

    # ✅ Display assistant response in chat message container
    st.chat_message("assistant", avatar=os.path.join(project_dir, "icons", "bot-icon.png")).markdown(answer)
    # ✅ Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # ✅ Refresh UI
    st.rerun()
