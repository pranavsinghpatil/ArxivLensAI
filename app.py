# app.py

from sentence_transformers import SentenceTransformer
import streamlit as st
import os
from qa_system import generate_answer_huggingface
from vector_store import search_faiss, load_faiss_index
from main import process_pdf
from utils import get_faiss_index_filename, CUSTOM_STYLES, PAGE_STYLES, EXTERNAL_DEPENDENCIES

# Fix GRPC error
os.environ["GRPC_DNS_RESOLVER"] = "ares"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token="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")

# Set up directories
project_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(project_dir, "temp")
faiss_indexes_dir = os.path.join(project_dir, "faiss_indexes")
extracted_images_dir = os.path.join(project_dir, "extracted_images")
tables_dir = os.path.join(project_dir, "extracted_tables")

# Ensure directories exist
for directory in [temp_dir, faiss_indexes_dir, extracted_images_dir, tables_dir]:
    os.makedirs(directory, exist_ok=True)

# Default Research Paper
default_paper_path = os.path.join(temp_dir, "Attention Is All You Need(default_research_paper).pdf")

# Streamlit UI setup
st.set_page_config(page_title="📄 AI-Powered Research Assistant", layout="wide")

# Apply styles
st.markdown(EXTERNAL_DEPENDENCIES + PAGE_STYLES + f"<style>{CUSTOM_STYLES}</style>", unsafe_allow_html=True)

# Display Fixed Header
st.markdown("<h3 style='text-align: center;'>🤖 AI-Powered Research Assistant</h3>", unsafe_allow_html=True)

# Sidebar - Multiple PDF Upload
st.sidebar.header("📄 Upload Research Papers")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Store uploaded papers
available_papers = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.toast(f"✅ {uploaded_file.name} uploaded!", icon="📄")

        try:
            faiss_index_filename = get_faiss_index_filename(pdf_path)
            faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)

            if not os.path.exists(faiss_index_path):
                process_pdf(pdf_path)

            available_papers[uploaded_file.name] = pdf_path
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

# Load Default Paper if No Uploads
if not available_papers and os.path.exists(default_paper_path):
    available_papers["Attention Is All You Need (Default)"] = default_paper_path
elif not available_papers:
    st.error("⚠️ Default research paper is missing! Please upload a file.")

# Dropdown to Select Research Papers
selected_papers = st.sidebar.multiselect(
    "📂 Select Research Papers",
    list(available_papers.keys()),
    default=list(available_papers.keys())
)
st.session_state.selected_papers = [available_papers[p] for p in selected_papers]

# Process selected papers
for pdf_path in st.session_state.selected_papers:
    try:
        faiss_index_filename = get_faiss_index_filename(pdf_path)
        faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)

        if not os.path.exists(faiss_index_path):
            process_pdf(pdf_path)
    except Exception as e:
        st.error(f"❌ Error processing {pdf_path}: {str(e)}")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat UI
chat_container = st.container()
chat_container.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display Chat History (Bottom to Top)
for chat in reversed(st.session_state.chat_history):
    if chat["role"] == "user":
        st.markdown(f'<div class="user-message">{chat["message"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{chat["message"]}</div>', unsafe_allow_html=True)

chat_container.markdown('</div>', unsafe_allow_html=True)

# Fixed Chat Input
st.markdown("""
    <div class="chat-input-wrapper">
        <div class="chat-input-container">
            <input type="text" class="chat-input" placeholder="Type your message..." id="chat-input">
            <button class="send-button">
                <i class="fas fa-paper-plane"></i>
            </button>
        </div>
    </div>
""", unsafe_allow_html=True)

# Hidden Streamlit input for handling the chat
query = st.text_input("", key="hidden_input", label_visibility="collapsed")

# Handle chat interactions
if query:
    st.session_state.chat_history.append({"role": "user", "message": query})

    all_retrieved_chunks = []
    for pdf_path in st.session_state.selected_papers:
        try:
            faiss_index, chunks = load_faiss_index(pdf_path)
            retrieved_chunks = search_faiss(query, faiss_index, embedding_model, chunks)
            all_retrieved_chunks.extend(retrieved_chunks)
        except Exception as e:
            st.error(f"❌ Error retrieving from {pdf_path}: {str(e)}")

    # Generate answer
    answer = generate_answer_huggingface(query, all_retrieved_chunks)
    st.session_state.chat_history.append({"role": "ai", "message": answer})

    # Refresh UI
    st.rerun()