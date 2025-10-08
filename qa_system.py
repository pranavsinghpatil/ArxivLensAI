from transformers import pipeline
import os
from vector_store import search_faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import google.generativeai as genai
import torch
import streamlit as st
from utils import GOOGLE_API_KEY, HUGGINGFACE_API_KEY

torch.classes.__path__ = []

# Initialize API keys lazily
google_api_key = GOOGLE_API_KEY
huggingface_api_key = HUGGINGFACE_API_KEY

def set_api_keys(gapi_key=None, hapi_key=None):
	"""Set API keys if provided."""
	global google_api_key, huggingface_api_key
	if gapi_key:
		google_api_key = gapi_key
	if hapi_key:
		huggingface_api_key = hapi_key

_gemini_model = None
_gemini_model_name = "gemini-pro"

def _normalize_gemini_model_name(model_name: str) -> str:
	"""Map various UI-friendly names to a server-supported model for v1beta generateContent."""
	if not isinstance(model_name, str) or not model_name.strip():
		return "gemini-pro"
	name = model_name.strip()
	# For v1beta generateContent, gemini-pro is the most broadly supported text model
	return "gemini-pro"

def set_gemini_model_name(model_name: str):
	"""Set the Gemini model name and reset cached model so it reloads next call."""
	global _gemini_model_name, _gemini_model
	_gemini_model_name = _normalize_gemini_model_name(model_name)
	_gemini_model = None

def get_gemini_model():
	global _gemini_model
	if _gemini_model is not None:
		return _gemini_model
	if not google_api_key:
		return None
	try:
		genai.configure(api_key=google_api_key)
		_gemini_model = genai.GenerativeModel(_gemini_model_name)
		return _gemini_model
	except Exception as e:
		print(f"[ERROR] Failed to initialize Gemini model: {e}")
		return None

_retrieval_pipeline = None
def get_retrieval_pipeline():
	global _retrieval_pipeline
	if _retrieval_pipeline is not None:
		return _retrieval_pipeline
	try:
		_retrieval_pipeline = pipeline(
			"question-answering",
			model="deepset/roberta-base-squad2",
			tokenizer="deepset/roberta-base-squad2"
		)
		return _retrieval_pipeline
	except Exception as e:
		print(f"⚠️ Retrieval pipeline failed to load: {e}")
		return None

def generate_research_answer(context, query, retrieved_text):
    """Uses Gemini to generate research-focused answers based on retrieved text."""
    
    print(f"[DEBUG] Generating research answer for query: {query}")
    print(f"[DEBUG] Context length: {len(context)}")
    print(f"[DEBUG] Retrieved text length: {len(retrieved_text)}")
    
    if "summary" in query.lower() or "summarize" in query.lower():
        # If the query is asking for a summary, generate a summary of the context
        prompt = f"""
        You are an AI research assistant. Provide a detailed and comprehensive summary of the following research paper context.
        Ensure the summary is at least 450 words long, covering all key points and providing in-depth analysis.

        **Context from research paper:**
        {context}

        **User's question:**
        {query}

        Provide a **detailed and clear** summary that includes:
        1. Main objectives and contributions
        2. Methodology and approach
        3. Key findings and results
        4. Implications and conclusions
        
        Base your response STRICTLY on the provided context.
        """
    else:
        # For other types of queries, generate a detailed, research-oriented response
        prompt = f"""
        You are an AI research assistant. Answer the user's query based STRICTLY on the research paper content.
        Ensure the response is at least 450 words long, providing detailed explanations, examples, and relevant context.

        **Context from research paper:**
        {context}

        **Most relevant extracted text:**
        {retrieved_text}

        **User's question:**
        {query}

        IMPORTANT RULES:
        1. ONLY use information from the provided paper context
        2. If you cannot find relevant information in the context, say so explicitly
        3. DO NOT make up or infer information not present in the context
        4. DO NOT use any external knowledge
        5. Cite specific sections or quotes from the paper when possible
        
        Provide a **detailed, research-oriented** response in clear language.
        """

    try:
        print("[DEBUG] Sending prompt to Gemini model...")
        gemini_model = get_gemini_model()
        if gemini_model is None:
            return "⚠️ Gemini model is not available. Please configure GOOGLE_API_KEY."
        response = gemini_model.generate_content(prompt)
        
        if response and response.text and response.text.strip():
            print("[DEBUG] Successfully generated response")
            return response.text.strip()
            
        print("[WARNING] Gemini model returned empty response")
        return "⚠️ Could not generate a response based on the paper content."
        
    except Exception as e:
        print(f"[ERROR] Failed to generate research answer: {e}")
        return f"⚠️ Error generating response: {str(e)}"

def generate_answer_huggingface(query, retrieved_chunks, memory=None, image_texts=None, table_texts=None, full_context=False):
    """
    Processes retrieved text, extracts relevant data, and generates a response using Gemini.
    """
    print(f"[DEBUG] Processing query: {query}")
    print(f"[DEBUG] Retrieved chunks: {len(retrieved_chunks)}")
    print(f"[DEBUG] Image texts: {len(image_texts) if image_texts else 0}")
    print(f"[DEBUG] Table texts: {len(table_texts) if table_texts else 0}")

    try:
        # ✅ Step 1: Handle full context request
        if full_context:
            context = " ".join(retrieved_chunks)  # Use all retrieved chunks
        else:
            if not retrieved_chunks:
                print("[WARNING] No relevant chunks found")
                return "I could not find relevant information in the paper to answer your question."

            # Use top chunks for focused context
            context = " ".join(retrieved_chunks[:3])  # Use top 3 most relevant chunks

        # ✅ Step 2: Add image and table context
        context_parts = [context]
        
        if image_texts:
            context_parts.append("\nInformation from images:")
            context_parts.extend(image_texts)
            
        if table_texts:
            context_parts.append("\nInformation from tables:")
            context_parts.extend(table_texts)
            
        combined_context = "\n".join(context_parts)
        print(f"[DEBUG] Combined context length: {len(combined_context)}")

        # ✅ Step 3: Get past context if available
        if memory:
            past_queries = [m["content"] for m in memory[-3:] if m["role"] == "user"]
            if past_queries:
                query_context = "Previous questions: " + "; ".join(past_queries)
                print(f"[DEBUG] Added query context: {query_context}")
                query = f"{query_context}\nCurrent question: {query}"

        # ✅ Step 4: Extract most relevant text
        retrieved_text = ""
        retrieval_pipeline = get_retrieval_pipeline()
        if retrieval_pipeline:
            try:
                ret_result = retrieval_pipeline(question=query, context=combined_context)
                retrieved_text = ret_result.get('answer', "")
                print(f"[DEBUG] Retrieved specific text: {retrieved_text[:100]}...")
            except Exception as e:
                print(f"[WARNING] Retrieval pipeline error: {e}")

        # ✅ Step 5: Generate research-based answer
        print("[DEBUG] Generating final answer...")
        answer = generate_research_answer(combined_context, query, retrieved_text)

        # ✅ Step 6: Ensure comprehensive response
        if len(answer.split()) < 450:
            print("[DEBUG] Answer too short, generating enriched response...")
            additional_context = "To provide a more comprehensive response, let's explore additional insights."
            enriched_answer = generate_research_answer(combined_context, query, retrieved_text + " " + additional_context)
            if len(enriched_answer.split()) > 450:
                answer = enriched_answer

        print("[DEBUG] Successfully generated answer")
        return answer

    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        return f"⚠️ Error generating answer: {str(e)}"
