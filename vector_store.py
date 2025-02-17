from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the SBERT model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  

def generate_embeddings(text_chunks):
    """Converts text chunks into vector embeddings."""
    return model.encode(text_chunks, convert_to_numpy=True)

def build_faiss_index(text_chunks):
    """Creates a FAISS search index for fast retrieval."""
    embeddings = generate_embeddings(text_chunks)
    dimension = embeddings.shape[1]  # Embedding size
    index = faiss.IndexFlatL2(dimension)  # L2 Distance Search
    index.add(embeddings)  # Store embeddings
    return index, embeddings
