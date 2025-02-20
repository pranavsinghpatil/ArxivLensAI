import faiss
import pickle
import sys
import os
import shutil
from extract_text import extract_text_from_pdf, extract_tables_from_pdf, extract_images_from_pdf
# from vector_store import build_faiss_index
from utils import get_faiss_index_filename, get_chunks_filename

# Ensure necessary directories exist
project_dir = os.path.dirname(os.path.abspath(__file__))
faiss_indexes_dir = os.path.join(project_dir, "faiss_indexes")
extracted_images_dir = os.path.join(project_dir, "extracted_images")
os.makedirs(faiss_indexes_dir, exist_ok=True)
os.makedirs(extracted_images_dir, exist_ok=True)

def process_pdf(pdf_path, force_reprocess=False):
    """Processes a PDF, extracts text, and builds a FAISS index."""
    
    from vector_store import build_faiss_index  # ✅ Import inside function to avoid circular import
    
    faiss_index_filename = get_faiss_index_filename(pdf_path)
    faiss_index_path = os.path.join("faiss_indexes", faiss_index_filename)
    
    if not force_reprocess and os.path.exists(faiss_index_path):
        print(f"[INFO] FAISS index already exists: {faiss_index_path}. Skipping index building.")
        return  

    print("[INFO] Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("[ERROR] No valid text extracted from the PDF.")
        return
    text_chunks = text.split(". ")

    print("[INFO] Building FAISS index...")
    faiss_index, embeddings, chunks = build_faiss_index(text_chunks, pdf_path)

    with open(f"faiss_indexes/chunks_{faiss_index_filename}.pkl", "wb") as f:
        pickle.dump(chunks, f)

    faiss.write_index(faiss_index, faiss_index_path)

    print(f"[SUCCESS] FAISS index saved to {faiss_index_path}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Please provide a PDF file path.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_pdf(pdf_path)
