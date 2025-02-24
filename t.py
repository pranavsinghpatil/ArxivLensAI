# qa_system.py
from transformers import pipeline
import os
from vector_store import search_faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import google.generativeai as genai
import torch
import streamlit as st
from utils import GOOGLE_API_KEY, HUGGINGFACE_API_KEY

# Initialize API keys from config or use placeholders
google_api_key = GOOGLE_API_KEY
huggingface_api_key = HUGGINGFACE_API_KEY

def set_api_keys(gapi_key=None, hapi_key=None):
    """Set API keys if provided."""
    global google_api_key, huggingface_api_key
    if gapi_key:
        google_api_key = gapi_key
    if hapi_key:
        huggingface_api_key = hapi_key

# 🔹 Step 1: Securely Configure Gemini API Key
if google_api_key:
    genai.configure(api_key=google_api_key)

# 🔹 Step 2: Initialize Gemini Model
gemini_model = genai.GenerativeModel("gemini-pro") if google_api_key else None

# ✅ Load Hugging Face Models
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# ✅ Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ✅ Hugging Face QA Pipelines
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ✅ Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token=huggingface_api_key) if huggingface_api_key else None

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

def generate_research_answer(context, query, retrieved_text):
    """Uses Gemini to generate research-focused answers based on retrieved text."""
    if "summary" in query.lower() or "summarize" in query.lower():
        # If the query is asking for a summary, generate a summary of the context
        prompt = f"""
        You are an AI research assistant. Provide a detailed and comprehensive summary of the following research paper context.
        Ensure the summary is at least 450 words long, covering all key points and providing in-depth analysis.

        **Context from research paper:**
        {context}

        **User's question:**
        {query}

        Provide a **detailed and clear** summary.
        """
    else:
        # For other types of queries, generate a detailed, research-oriented response
        prompt = f"""
        You are an AI research assistant. Answer the user's query based on the research paper.
        Ensure the response is at least 450 words long, providing detailed explanations, examples, and relevant context.

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
def generate_answer_huggingface(query, retrieved_chunks, memory, image_texts, table_texts, full_context=False):
    """
    Processes retrieved text, extracts relevant data, and generates a response using Gemini.

    Args:
        query (str): User's query.
        retrieved_chunks (list): Retrieved text chunks from FAISS.
        memory (list): Chat memory for past user interactions.
        image_texts (list): Text extracted from images.
        table_texts (list): Text extracted from tables.
        full_context (bool, optional): If True, uses full document context. Defaults to False.

    Returns:
        str: Generated response.
    """

    # ✅ Step 1: Handle full context request
    if full_context:
        context = " ".join(retrieved_chunks)  # Use all retrieved chunks
    else:
        if not retrieved_chunks:
            # ✅ Step 2: Fallback research-based response if no relevant text found
            research_fallback_response = generate_research_answer(
                context="No direct references found in the selected research papers.",
                query=query,
                retrieved_text="However, based on research principles and general AI knowledge, here is an answer:"
            )
            return research_fallback_response if research_fallback_response.strip() else f"⚠️ The research papers do not contain information on {query}."

        # ✅ Step 3: Combine retrieved chunks into context
        context = " ".join(retrieved_chunks)

    # ✅ Step 4: Retrieve past 3 user queries to maintain conversation flow
    past_context = " ".join([m["content"] for m in memory[-3:] if m["role"] == "user"])

    # ✅ Step 5: Modify query with past context
    query_with_memory = f"Considering our past discussion: {past_context}. Now, {query}" if past_context else query

    # ✅ Step 6: Combine text from images and tables with the main context
    combined_context = f"{context} {' '.join(image_texts)} {' '.join(table_texts)}"

    # ✅ Step 7: Extract Most Relevant Text using Retrieval Model (if available)
    retrieved_text = ""
    if retrieval_pipeline:
        try:
            ret_result = retrieval_pipeline(question=query_with_memory, context=combined_context)
            retrieved_text = ret_result.get('answer', "")
        except Exception as e:
            print(f"⚠️ Retrieval pipeline error: {e}")

    # ✅ Step 8: Generate Research-Based Answer
    answer = generate_research_answer(combined_context, query_with_memory, retrieved_text)

    # ✅ Step 9: Ensure answer has enough depth (≥ 450 words)
    if len(answer.split()) < 450:
        additional_context = "To provide a more comprehensive response, let's explore additional insights."
        enriched_answer = generate_research_answer(combined_context, query, retrieved_text + " " + additional_context)
        if len(enriched_answer.split()) > 450:
            answer = enriched_answer

    return answer