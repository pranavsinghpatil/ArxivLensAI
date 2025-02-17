import faiss
from extract_text import extract_text_from_pdf, clean_text
from vector_store import build_faiss_index, model

# Load and clean text
pdf_path = "sample.pdf"
raw_text = extract_text_from_pdf(pdf_path)
cleaned_text = clean_text(raw_text)

# Split text into chunks
text_chunks = cleaned_text.split(". ")

# Build FAISS index
#
faiss_index, embeddings, chunks = build_faiss_index(model, text_chunks)  # ✅ Pass 'model'
print("Type of chunks:", type(chunks))
print("Length of chunks:", len(chunks) if isinstance(chunks, list) else "Not a list")
print("First 5 chunks:", chunks[:5] if isinstance(chunks, list) else "Not a list")

print("FAISS index built successfully!")

faiss.write_index(faiss_index, "faiss_index.index")
print("FAISS index saved to faiss_index.index.")