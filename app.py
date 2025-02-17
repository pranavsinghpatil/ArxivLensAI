from sentence_transformers import SentenceTransformer
import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
from qa_system import generate_answer_huggingface, model , embedding_model
from vector_store import search_faiss
import subprocess
import os

# Streamlit UI setup
st.title("📄 AI-Powered Research Assistant")
st.sidebar.header("Upload Your Research Paper")

# PDF Upload Section
uploaded_file = st.sidebar.file_uploader("Upload a Research Paper (PDF)", type="pdf")

if uploaded_file:
    # Save the uploaded PDF temporarily
    pdf_path = os.path.join("temp", uploaded_file.name)  # Save in a "temp" folder
    os.makedirs("temp", exist_ok=True)  # Ensure "temp" folder exists

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success("✅ PDF Uploaded Successfully!")

    # Trigger processing in main.py to create FAISS index
    subprocess.run(["python", "main.py", pdf_path]) # Calls main.py to process and build FAISS index

    st.success("✅ PDF processed & indexed! You can now ask questions.")

# Query Input
query = st.text_input("🔎 Ask a question about the research paper:")

if st.button("Get Answer") and query:
    try:
        # Load the FAISS index for querying
        with open("faiss_index.index", "rb") as f:
            faiss_index = pickle.load(f)
        
        with open("chunks.pkl", "rb") as f:  # Assuming you saved chunks in "chunks.pkl"
            chunks = pickle.load(f)
        
        # Retrieve relevant chunks from FAISS index
        retrieved_chunks = search_faiss(query, faiss_index , chunks)

        # Generate answer using HuggingFace model
        answer = generate_answer_huggingface(query, retrieved_chunks)

        st.subheader("🔍 Answer:")
        st.write(answer)
    except FileNotFoundError:
        st.error("FAISS index not found. Please upload a PDF first.")

