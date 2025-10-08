from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import hashlib
import os
from utils import get_faiss_index_filename, expand_query, GOOGLE_API_KEY, HUGGINGFACE_API_KEY
import streamlit as st
import torch
import torch.nn.functional as F

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

_embedding_model = None
def initialize_embedding_model():
    """Initialize the sentence transformer model with proper error handling."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        # Allow anonymous loading if token is not provided (for local runs)
        if huggingface_api_key:
            _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=huggingface_api_key)
        else:
            _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[SUCCESS] Successfully initialized embedding model")
        return _embedding_model
    except Exception as e:
        print(f"[ERROR] Failed to initialize embedding model: {e}")
        return None

def get_embedding_model():
    """Return initialized embedding model if available."""
    return _embedding_model

project_dir = os.path.dirname(os.path.abspath(__file__))
faiss_indexes_dir = os.path.join(project_dir, "faiss_indexes")
os.makedirs(faiss_indexes_dir, exist_ok=True)

def encode_chunks_parallel(text_chunks):
    """Encodes text chunks efficiently in batches."""
    if _embedding_model is None:
        print("[ERROR] Embedding model is not initialized")
        return None
        
    try:
        print(f"[DEBUG] Encoding {len(text_chunks)} chunks in batches")
        embeddings = _embedding_model.encode(text_chunks, batch_size=16, show_progress_bar=True)
        print(f"[DEBUG] Successfully encoded chunks to shape {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"[ERROR] Failed to encode chunks: {e}")
        return None

def encode_query(query, model):
    """Encode a query string into an embedding vector."""
    try:
        # Normalize and clean query
        query = query.strip().lower()
        if not query:
            return None
            
        # Get embedding using sentence transformer directly
        embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        if embedding is None or embedding.size == 0:
            print("[ERROR] Failed to generate query embedding")
            return None
            
        return embedding
        
    except Exception as e:
        print(f"[ERROR] Query encoding failed: {e}")
        return None

def build_faiss_index(text_chunks, pdf_path):
    """Build a FAISS index from text chunks."""
    print("[DEBUG] Building FAISS index...")
    print(f"[DEBUG] Number of chunks: {len(text_chunks)}")
    
    try:
        # Encode chunks
        print("[DEBUG] Encoding text chunks...")
        embeddings = encode_chunks_parallel(text_chunks)
        if embeddings is None or len(embeddings) == 0:
            print("[ERROR] Failed to generate embeddings")
            return None, None, None
            
        print(f"[DEBUG] Generated embeddings with shape: {embeddings.shape}")
        
        # Create and train index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        print("[DEBUG] Training FAISS index...")
        if len(text_chunks) > 1:
            index.train(embeddings)
        
        print("[DEBUG] Adding vectors to index...")
        index.add(embeddings)
        
        print("[DEBUG] Saving files...")
        print(f"[SUCCESS] FAISS index saved to {os.path.join(faiss_indexes_dir, get_faiss_index_filename(pdf_path))}")
        print(f"[SUCCESS] Chunks saved to {os.path.join(faiss_indexes_dir, f'chunks_{get_faiss_index_filename(pdf_path)}.pkl')}")
        
        return index, embeddings, text_chunks
        
    except Exception as e:
        print(f"[ERROR] Failed to build FAISS index: {e}")
        return None, None, None

def search_faiss(query, index, embedding_model, chunks, memory=None, k=5):
    """Search for relevant chunks using FAISS."""
    print(f"[DEBUG] Searching for query: {query}")
    print(f"[DEBUG] Total available chunks: {len(chunks)}")
    
    try:
        # Encode query
        query_embedding = encode_query(query, embedding_model)
        if query_embedding is None:
            print("[ERROR] Failed to encode query")
            return None
            
        # Search index
        D, I = index.search(query_embedding.reshape(1, -1), k)
        print(f"[DEBUG] Search results - distances: {D[0]}, indices: {I[0]}")
        
        # Get relevant chunks
        relevant_chunks = []
        for i, idx in enumerate(I[0]):
            if idx < len(chunks):  # Validate index
                chunk = chunks[idx]
                distance = D[0][i]
                print(f"[DEBUG] Chunk {i+1} (distance={distance:.4f}):")
                print(f"[DEBUG] {chunk[:200]}...")
                relevant_chunks.append(chunk)
            else:
                print(f"[WARNING] Invalid chunk index: {idx}")
                
        if not relevant_chunks:
            print("[WARNING] No relevant chunks found")
            return None
            
        print(f"[DEBUG] Found {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
        
    except Exception as e:
        print(f"[ERROR] FAISS search failed: {e}")
        return None

faiss_index_cache = {}  # Cache to store loaded indexes

def load_faiss_index(pdf_path):
    """Loads FAISS index and text chunks based on PDF path."""
    base_filename = get_faiss_index_filename(pdf_path)
    faiss_index_filename = os.path.join(faiss_indexes_dir, base_filename)
    chunks_filename = os.path.join(faiss_indexes_dir, f"chunks_{base_filename}.pkl")

    if not os.path.exists(faiss_index_filename):
        raise FileNotFoundError(f" FAISS index file not found: {faiss_index_filename}")
    if not os.path.exists(chunks_filename):
        raise FileNotFoundError(f" Chunks file not found: {chunks_filename}")

    index = faiss.read_index(faiss_index_filename)
    with open(chunks_filename, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks
