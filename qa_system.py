from transformers import pipeline
import os
from vector_store import search_faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ✅ Load the embedding model separately
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Hugging Face text generation model
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")  

def generate_answer_huggingface(query, retrieved_chunks):
    print("Retrieved Chunks Type:", type(retrieved_chunks))
    print("Retrieved Chunks Value:", retrieved_chunks)

    context = " ".join(retrieved_chunks)  # Combine retrieved text
    prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    # Tokenize the prompt for the model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate the output
    output = model.generate(**inputs)

    # Decode the generated answer
    return tokenizer.decode(output[0], skip_special_tokens=True)

