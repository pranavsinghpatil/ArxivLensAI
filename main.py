# from extract_text import extract_text_from_pdf, clean_text
from vector_store import build_faiss_index, generate_embeddings

# Load and clean text
# pdf_path = "sample.pdf"
# raw_text = extract_text_from_pdf(pdf_path)
# cleaned_text = clean_text(raw_text)

# Split text into chunks (important for retrieval)
# text_chunks = cleaned_text.split(". ")  # Simple sentence-level split

# Build FAISS index
# faiss_index, embeddings = build_faiss_index(text_chunks)
# print("FAISS index built successfully!")
