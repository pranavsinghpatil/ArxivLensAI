# qa_system.py
from transformers import pipeline
import os
from vector_store import search_faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import google.generativeai as genai
import torch

# 🔹 Step 1: Securely Configure Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Load from environment variable
if not GOOGLE_API_KEY:
    # raise ValueError("⚠️ Missing GOOGLE_API_KEY! Set it in environment variables.")
    GOOGLE_API_KEY = "AIzaSyAYqEVojzmSLv101fVPvEzDHLhpuR7SYso"
genai.configure(api_key=GOOGLE_API_KEY)

# 🔹 Step 2: Initialize Gemini Model
gemini_model = genai.GenerativeModel("gemini-pro")

# ✅ Load Embedding Model (Correcting `token` issue)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token="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")  

# ✅ Load Hugging Face Models
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# ✅ Use GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model.to(device)

# ✅ Hugging Face QA Pipelines
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HFr_API_TOKEN")  # Secure Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ✅ Robust Retrieval Pipeline with Error Handling
try:
    retrieval_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2"
    )
except Exception as e:
    print(f"⚠️ Retrieval pipeline failed to load: {e}")
    retrieval_pipeline = None  # Ensure code doesn’t break

# 🔹 Research AI Answer Generation using Gemini
def generate_research_answer(context, query, retrieved_text):
    """Uses Gemini to generate research-focused answers based on retrieved text."""
    prompt = f"""
    You are an AI research assistant. Answer the user's query based on the research paper.
    
    **Context from research paper:**
    {context}

    **Most relevant extracted text:**
    {retrieved_text}

    **User's question:**
    {query}

    Provide a **detailed, research-oriented** response in clear language.
    """
    try:
        response = gemini_model.generate_content(prompt)
        if response.text.strip():
            return response.text.strip()
        return "⚠️ Gemini model returned an empty response."
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# 🔹 Hugging Face Answer Generation
def generate_answer_huggingface(query, retrieved_chunks):
    """Processes retrieved text, extracts relevant data, and generates a response using Gemini."""
    if not retrieved_chunks:
        return generate_research_answer(
            context="The research paper does not directly mention this topic.",
            query=query,
            retrieved_text="No direct references, but answering based on research principles."
        )


    # ✅ Combine Retrieved Chunks into Context
    context = " ".join(retrieved_chunks)
    print("\n[INFO] Retrieved Context:\n", context, "\n")

    # ✅ Extract Most Relevant Text using Retrieval Model
    if retrieval_pipeline:
        try:
            ret_result = retrieval_pipeline(question=query, context=context)
            retrieved_text = ret_result['answer']
            print("\n[INFO] Retrieved Text:\n", retrieved_text, "\n")
        except Exception as e:
            print(f"⚠️ Retrieval pipeline error: {e}")
            retrieved_text = ""
    else:
        retrieved_text = ""

    # ✅ Generate Research-Based Answer
    answer = generate_research_answer(context, query, retrieved_text)
    print("\n[INFO] Generated Answer:\n", answer, "\n")
    
    return answer
