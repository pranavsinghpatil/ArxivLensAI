import faiss
from extract_text import extract_text_from_pdf, clean_text
from vector_store import build_faiss_index
import pickle

def process_pdf(pdf_path):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    # Split text into chunks
    text_chunks = text.split(". ")

    # Build FAISS index
    faiss_index, embeddings, chunks = build_faiss_index(text_chunks)
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # Save FAISS index to file
    with open("faiss_index.index", "wb") as f:
        pickle.dump(faiss_index, f)
    
    print("FAISS index saved to faiss_index.index.")

if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1]  # Get the pdf path passed from app.py
    process_pdf(pdf_path)
