from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from concurrent.futures import ThreadPoolExecutor
import faiss
import numpy as np
import pickle

# ✅ Load SentenceTransformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_chunks_parallel(text_chunks):
    """Encodes text chunks in parallel for better speed."""
    with ThreadPoolExecutor(max_workers=4) as executor:  # 4 parallel threads
        embeddings = list(executor.map(model.encode, text_chunks))
    return np.array(embeddings)

def build_faiss_index(text_chunks):
    """Creates and saves FAISS index faster using parallel encoding."""
    embeddings = encode_chunks_parallel(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # ✅ Save FAISS index
    faiss.write_index(index, "faiss_index.index")

    # ✅ Save chunks
    with open("chunks.pkl", "wb") as f:
        pickle.dump(text_chunks, f)

    return index, embeddings, text_chunks

def load_faiss_index():
    """Loads FAISS index and text chunks from file."""
    index = faiss.read_index("faiss_index.index")  # ✅ Load FAISS index
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)  # ✅ Load chunks
    return index, chunks

def search_faiss(faiss_index, query, chunks):
    """Search for the most relevant chunks in FAISS index."""
    # ✅ Use SentenceTransformer for encoding
    query_embedding = embedding_model.encode([query])  # Use the SentenceTransformer model
    query_embedding = np.array(query_embedding)
    # Search FAISS index
    _, indices = faiss_index.search(query_embedding, top_k=3)

    return [chunks[i] for i in indices[0] if i < len(chunks)]  # ✅ Return actual retrieved chunks
