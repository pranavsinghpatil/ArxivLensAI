from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import hashlib
import os
from utils import get_faiss_index_filename, expand_query, GOOGLE_API_KEY, HUGGINGFACE_API_KEY
import streamlit as st

# Initialize API keys from config or use placeholders
google_api_key = GOOGLE_API_KEY
huggingface_api_key = HUGGINGFACE_API_KEY

def setv_api_keys(gapi_key=None, hapi_key=None):
    """Set API keys if provided."""
    global google_api_key, huggingface_api_key
    if gapi_key:
        google_api_key = gapi_key
    if hapi_key:
        huggingface_api_key = hapi_key

# Load Model Once
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=huggingface_api_key) if huggingface_api_key else None

project_dir = os.path.dirname(os.path.abspath(__file__))
faiss_indexes_dir = os.path.join(project_dir, "faiss_indexes")
os.makedirs(faiss_indexes_dir, exist_ok=True)

def encode_chunks_parallel(text_chunks):
    """Encodes text chunks efficiently in batches."""
    return embedding_model.encode(text_chunks, batch_size=16, show_progress_bar=True)

def build_faiss_index(text_chunks, pdf_path):
    """Creates and saves FAISS index efficiently with unique filenames."""
    embeddings = encode_chunks_parallel(text_chunks)

    # Use HNSW only if supported (IVFFlat does NOT support it)
    quantizer = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], 100)  # 100 clusters

    index.train(embeddings)  # Train index with embeddings
    index.add(embeddings)

    # Generate unique FAISS filename
    base_filename = get_faiss_index_filename(pdf_path)
    faiss_index_filename = os.path.join(faiss_indexes_dir, base_filename)

    # Save FAISS index
    faiss.write_index(index, faiss_index_filename)

    # Save text chunks
    chunks_filename = os.path.join(faiss_indexes_dir, f"chunks_{base_filename}.pkl")
    with open(chunks_filename, "wb") as f:
        pickle.dump(text_chunks, f)

    print(f"✅ FAISS index saved to {faiss_index_filename}.")
    print(f"✅ Chunks saved to {chunks_filename}.")

    return index, embeddings, text_chunks

faiss_index_cache = {}  # Cache to store loaded indexes

def load_faiss_index(pdf_path):
    """Loads FAISS index and text chunks based on PDF path."""
    base_filename = get_faiss_index_filename(pdf_path)
    faiss_index_filename = os.path.join(faiss_indexes_dir, base_filename)
    chunks_filename = os.path.join(faiss_indexes_dir, f"chunks_{base_filename}.pkl")

    if not os.path.exists(faiss_index_filename):
        raise FileNotFoundError(f"⚠️ FAISS index file not found: {faiss_index_filename}")
    if not os.path.exists(chunks_filename):
        raise FileNotFoundError(f"⚠️ Chunks file not found: {chunks_filename}")

    index = faiss.read_index(faiss_index_filename)
    with open(chunks_filename, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks

def search_faiss(query, faiss_index, embedding_model, chunks, memory, k=5):
    """
    Searches FAISS index for the most relevant chunks based on the query.
    Includes query expansion and threshold-based filtering.

    Args:
        query (str): User's search query.
        faiss_index (faiss.Index): The FAISS index for fast retrieval.
        embedding_model (SentenceTransformer): The sentence embedding model.
        chunks (list): List of text chunks.
        memory (list): Chat history for query expansion.
        k (int, optional): Number of top results to retrieve. Defaults to 5.

    Returns:
        list: The most relevant retrieved text chunks.
    """
    if faiss_index is None:
        raise ValueError("❌ FAISS index is not loaded. Please process the PDF first.")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("❌ Query must be a non-empty string.")

    # Expand query using past user interactions
    expanded_query = expand_query(query, memory)

    # Encode the expanded query
    query_embedding = embedding_model.encode([expanded_query], convert_to_numpy=True).astype("float32")

    # Perform the FAISS search
    D, I = faiss_index.search(query_embedding, k)

    # Apply dynamic thresholding to filter low-confidence results
    if D.size == 0 or I.size == 0:
        return ["I couldn't find relevant information."]

    # Compute the threshold dynamically (percentile-based filtering)
    threshold = np.percentile(D[0], 75)  # Consider the top 25% of results
    filtered_chunks = [chunks[i] for i, d in zip(I[0], D[0]) if i < len(chunks) and d < threshold]

    return filtered_chunks if filtered_chunks else ["I couldn't find relevant information."]
