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

# ✅ Streamlit UI setup
st.set_page_config(page_title="📄 AI-Powered Research Assistant", layout="wide")

st.markdown("""
    <style>
        .chat-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            background-color: #0084FF;
            color: white;
            padding: 10px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 60%;
        }
        .ai-message {
            background-color: #E5E5EA;
            color: black;
            padding: 10px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 60%;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Powered Research Assistant")

# 📌 Sidebar - Research Paper Upload
st.sidebar.header("📄 Upload Your Research Paper")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False

# 📑 Process the uploaded PDF
if uploaded_file:
    pdf_path = os.path.join(temp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("✅ PDF Uploaded Successfully!")

    # ✅ Ensure FAISS index is built
    try:
        faiss_index_filename = get_faiss_index_filename(pdf_path)
        faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)
    
        if not os.path.exists(faiss_index_path):
            st.info("🔄 Processing PDF and building FAISS index... Please wait.")
            process_pdf(pdf_path)  # ✅ Automatically process the PDF if index is missing
            st.success("✅ PDF processed & indexed! You can now ask questions.")

        # ✅ Attempt to load FAISS index
        faiss_index, chunks = load_faiss_index(pdf_path)
        st.session_state.faiss_index = faiss_index
        st.session_state.chunks = chunks
        st.session_state.index_loaded = True

    except FileNotFoundError:
        st.error(f"⚠️ FAISS index or chunks file missing. Attempting to regenerate...")
        process_pdf(pdf_path)  # 🔄 Reprocess the PDF
        try:
            faiss_index, chunks = load_faiss_index(pdf_path)
            st.session_state.faiss_index = faiss_index
            st.session_state.chunks = chunks
            st.session_state.index_loaded = True
        except Exception as e:
            st.error(f"❌ FAISS index could not be loaded even after regeneration: {str(e)}")
            st.stop()


    # ✅ Expandable Research Paper Viewer
    with st.sidebar.expander("📜 View Research Paper Content", expanded=False):
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        st.download_button(label="📥 Download PDF", data=pdf_bytes, file_name=uploaded_file.name, mime="application/pdf")

    # ✅ Load Extracted Tables
    tables_file = os.path.join(tables_dir, f"tables_{get_faiss_index_filename(pdf_path)}.md")
    if os.path.exists(tables_file):
        st.subheader("📊 Extracted Tables from Research Paper")
        with open(tables_file, "r") as f:
            tables_content = f.read()
            st.markdown(f"```md\n{tables_content}\n```")

    # ✅ Load Extracted Images
    image_files = [f for f in os.listdir(extracted_images_dir) if f.startswith(os.path.basename(pdf_path))]
    if image_files:
        st.subheader("🖼 Extracted Images from Research Paper")
        for image_file in image_files:
            image_path = os.path.join(extracted_images_dir, image_file)
            st.image(image_path, caption=f"Extracted Image: {image_file}", use_column_width=True)

    # Load extracted tables
    tables_file = os.path.join(tables_dir, f"tables_{get_faiss_index_filename(pdf_path)}.pkl")
    if os.path.exists(tables_file):
        try:
            with open(tables_file, 'rb') as f:
                extracted_tables = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load tables: {str(e)}")
            extracted_tables = []

# 💬 Chatbot UI
st.subheader("💡 Ask a Question About the Research Paper")
query = st.chat_input("Type your question here...")

# Initialize extracted_tables
extracted_tables = []

if query:
    st.session_state.chat_history.append({"role": "user", "message": query})

    if not st.session_state.index_loaded:
        st.error("Please wait for the PDF to be processed.")
    else:
        retrieved_chunks = search_faiss(
            query, 
            st.session_state.faiss_index, 
            embedding_model, 
            st.session_state.chunks
        )

        # ✅ Check if tables contain relevant answers
        table_answers = []
        if os.path.exists(tables_file):
            with open(tables_file, "r") as f:
                tables_content = f.read()
                if query.lower() in tables_content.lower():
                    table_answers.append(f"📊 Relevant Table Data:\n\n{tables_content}")

        # ✅ Generate the final answer using Gemini or Hugging Face
        answer = generate_answer_huggingface(query, retrieved_chunks)
        st.session_state.chat_history.append({"role": "ai", "message": answer})

# 📌 Display Chat History
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-message'>{chat['message']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-message'>{chat['message']}</div>", unsafe_allow_html=True)

# Display tables section
if extracted_tables:
    st.subheader("📊 Extracted Tables")
    for idx, table in enumerate(extracted_tables):
        st.write(f"Table {idx+1}:")
        try:
            st.dataframe(pd.DataFrame(table))
        except Exception as e:
            st.error(f"Error displaying table {idx+1}: {str(e)}")
