from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(text_chunks):
    """Converts text chunks into vector embeddings."""
    return np.array(model.encode(text_chunks))  # No need for any changes here

def search_faiss(faiss_index, query, model, chunks, top_k=3):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")  # Or use the correct T5 tokenizer

    # Tokenize the query
    inputs = tokenizer(query, return_tensors="pt")
    
    # Use the model to generate a sequence
    with torch.no_grad():
        output = model.generate(**inputs)

    # Decode the generated output
    generated_query = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Generated Query: {generated_query}")
    
    # You can now use FAISS to retrieve the top K most relevant chunks based on the generated query
    # For now, let's use a simple embedding approach, since we're only generating a response:
    # This part can be extended with FAISS retrieval based on the output of T5.
    
    # Assuming faiss_index and chunks are pre-built, this is a placeholder for FAISS retrieval
    # Your FAISS retrieval logic will go here, based on either embeddings or similarity search.
    return chunks[:top_k]  # Placeholder for the top-k retrieved chunks


def build_faiss_index(model, text_chunks):
    # Convert each chunk into an embedding
    embeddings = np.array([model.encode(chunk) for chunk in text_chunks])  # ✅ Remove extra brackets
    
    # Ensure correct shape (N, D)
    if len(embeddings.shape) == 3:  # If shape is (N, 1, D), remove extra dimension
        embeddings = embeddings.squeeze(1)

    index = faiss.IndexFlatL2(embeddings.shape[1])  # D = embedding size
    index.add(embeddings)
    
    return index, embeddings, text_chunks