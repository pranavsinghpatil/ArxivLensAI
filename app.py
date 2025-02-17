from qa_system import generate_answer_huggingface
from main import faiss_index, chunks
from vector_store import search_faiss , model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define your query here
query = "Why the convergence of AI and hardware should enable generalizable robotics?"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")  # Or use another seq2seq model like BART
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Define your query
query = "What is deep learning?"

# Retrieve relevant chunks using FAISS
retrieved_chunks = search_faiss(faiss_index, query, model, chunks, top_k=3)

# Generate the answer using the Hugging Face model and retrieved chunks
answer = generate_answer_huggingface(query, retrieved_chunks, model, tokenizer)

print("Answer:", answer)
