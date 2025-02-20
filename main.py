# main.py
import faiss
import pickle
import sys
from extract_text import extract_text_from_pdf, clean_text
from vector_store import build_faiss_index
import hashlib
import os 
from utils import get_faiss_index_filename

def process_pdf(pdf_path, force_reprocess=False):
    """Processes a PDF, extracts text, and builds a FAISS index.
    
    - If the FAISS index **already exists**, it skips processing unless `force_reprocess=True`.
    - Saves **both the FAISS index and extracted text chunks** for retrieval.
    """
    # ✅ Generate a unique filename for FAISS index
    faiss_index_filename = get_faiss_index_filename(pdf_path)
    faiss_index_path = os.path.join("faiss_indexes", faiss_index_filename)
    chunks_path = os.path.join("faiss_indexes", f"chunks_{faiss_index_filename}.pkl")

    # ✅ Check if FAISS index exists (to avoid redundant processing)
    if not force_reprocess and os.path.exists(faiss_index_path):
        print(f"\n[INFO] FAISS index already exists: {faiss_index_path}. Skipping index building.\n")
        return  

    print("\n[INFO] Extracting text from PDF...\n")
    text = extract_text_from_pdf(pdf_path)

    # ✅ Handle empty or invalid text extraction
    if not text.strip():
        print("\n[ERROR] No valid text extracted from the PDF.\n")
        return

    # ✅ Split extracted text into chunks
    text_chunks = text.split(". ")

    print("\n[INFO] Building FAISS index...\n")
    faiss_index, embeddings, chunks = build_faiss_index(text_chunks, pdf_path)

    # ✅ Save text chunks
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # ✅ Save FAISS index
    faiss.write_index(faiss_index, faiss_index_path)

    print(f"\n[SUCCESS] FAISS index saved to {faiss_index_path}.\n")
    print(f"\n[SUCCESS] Chunks saved to {chunks_path}.\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Please provide a PDF file path.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_pdf(pdf_path)

