# utils.py
import hashlib
import os

def get_faiss_index_filename(pdf_path):
    """Generates a unique filename for FAISS index based on the PDF path."""
    pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    return f"faiss_index_{pdf_hash}.index"

def get_chunks_filename(pdf_path):
    """Generates a unique filename for text chunks based on the PDF path."""
    pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    return f"chunks_{pdf_hash}.pkl"
 