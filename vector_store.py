# vector_store.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import hashlib
from utils import get_faiss_index_filename
import os
# ✅ Load Model Once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token="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")

project_dir = os.path.dirname(os.path.abspath(__file__))
faiss_indexes_dir = os.path.join(project_dir, "faiss_indexes")
os.makedirs(faiss_indexes_dir, exist_ok=True)

def encode_chunks_parallel(text_chunks):
    """Encodes text chunks efficiently in batches (no multiprocessing)."""
    return embedding_model.encode(text_chunks, batch_size=16, show_progress_bar=True)

def get_faiss_index_filename(pdf_path):
    """Generates a unique filename for the FAISS index based on the PDF path."""
    pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    return f"faiss_index_{pdf_hash}.index"

def build_faiss_index(text_chunks, pdf_path):
    """Creates and saves FAISS index efficiently with unique filenames."""
    embeddings = encode_chunks_parallel(text_chunks)

    # ✅ Use HNSW only if supported (IVFFlat does NOT support it)
    quantizer = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], 100)  # 100 clusters

    index.train(embeddings)  # Train index with embeddings
    index.add(embeddings)

    # ✅ Generate unique FAISS filename
    base_filename = get_faiss_index_filename(pdf_path)
    faiss_index_filename = os.path.join(faiss_indexes_dir, f"{base_filename}.index")

    # ✅ Save FAISS index
    faiss.write_index(index, faiss_index_filename)

    # ✅ Save text chunks
    chunks_filename = os.path.join(faiss_indexes_dir, f"chunks_{base_filename}.pkl")
    with open(chunks_filename, "wb") as f:
        pickle.dump(text_chunks, f)

    print(f"✅ FAISS index saved to {faiss_index_filename}.")
    print(f"✅ Chunks saved to {chunks_filename}.")

    return index, embeddings, text_chunks


faiss_index_cache = {}  # ✅ Cache to store loaded indexes

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


def search_faiss(query, faiss_index, embedding_model, chunks):
    """Searches FAISS index for the most relevant chunks based on the query."""
    
    # Ensure FAISS index is valid
    if faiss_index is None:
        raise ValueError("FAISS index is not loaded. Please process the PDF first.")
    
    # Check if query is a string and encode it properly
    if isinstance(query, str):  # Ensure query is a string
        query_embedding = embedding_model.encode([query])  # Pass it as a list of 1 string
    else:
        raise ValueError("Query must be a string.")
    
    # Convert to NumPy format if necessary
    query_embedding = np.array(query_embedding).astype("float32")

    # Perform the FAISS search
    D, I = faiss_index.search(query_embedding, k=5)  # Retrieve top 5 results

    # Filter out low-confidence results (if FAISS distance is too high)
    threshold = np.percentile(D[0], 75)  # Increase to 75th percentile
    filtered_chunks = [chunks[i] for i, d in zip(I[0], D[0]) if i < len(chunks) and d < threshold]

    if not filtered_chunks:  # If no good results, return a fallback message
        return ["I couldn't find relevant information."]
    
    return filtered_chunks
