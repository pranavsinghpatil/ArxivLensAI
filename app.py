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
st.title("📄 AI-Powered Research Assistant")
st.sidebar.header("Upload Your Research Paper")

# ✅ PDF Upload Section
uploaded_file = st.sidebar.file_uploader("Upload a Research Paper (PDF)", type="pdf")

if uploaded_file:
    # ✅ Save the uploaded PDF temporarily
    pdf_path = os.path.join(temp_dir, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("✅ PDF Uploaded Successfully!")

    # ✅ Ensure FAISS index is built
    try:
        faiss_index_filename = get_faiss_index_filename(pdf_path)
        faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)

        if not os.path.exists(faiss_index_path):
            from main import process_pdf
            st.info("🔄 Processing PDF and building FAISS index... Please wait.")
            process_pdf(pdf_path)  # ✅ Automatically process the PDF if index is missing
            st.success("✅ PDF processed & indexed! You can now ask questions.")

        # ✅ Load FAISS index and chunks
        try:
            faiss_index, chunks = load_faiss_index(pdf_path)
            st.session_state.faiss_index = faiss_index
            st.session_state.chunks = chunks
            st.session_state.index_loaded = True
        except FileNotFoundError:
            st.error("⚠️ FAISS index could not be loaded. Please re-upload the PDF.")
            st.stop()

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.stop()

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

# ✅ Query Input
query = st.text_input("🔎 Ask a question about the research paper:")

if st.button("Get Answer", key="get_answer_button") and query:
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
    elif not st.session_state.get("index_loaded", False):
        st.error("Please wait for the PDF to be processed.")
    else:
        # ✅ Search FAISS index
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

        # ✅ Display the retrieved answer
        st.subheader("🔍 Answer:")
        st.write(answer)

        # ✅ Display table answers if relevant
        if table_answers:
            st.subheader("📊 Additional Relevant Table Data:")
            st.markdown("\n\n".join(table_answers))
