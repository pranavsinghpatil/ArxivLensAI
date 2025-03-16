# ArxivLensAI API Reference

This document details the APIs used within **ArxivLensAI**—an AI-powered personal research assistant. It explains how to configure and obtain the necessary API keys for external services, and describes the internal endpoints and functions that drive the application.

---

## 1. External APIs

### 1.1 Hugging Face Transformers API

**Purpose:**  
Generates answers and supports various natural language processing (NLP) tasks such as question answering and text generation.

**Usage:**  
The Hugging Face pipeline is used internally for these tasks. For example:
```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
answer = qa_pipeline(
    question="What is deep learning?", 
    context="Your research context here"
)["answer"]
```

**API Key Setup:**

Although many Hugging Face models are accessible without an API key, premium models or higher usage limits require an API key.
To obtain an API key:
1. Visit Hugging Face.
2. Create or log into your account.
3. Navigate to your Access Tokens settings.
4. Generate a new token.
5. Set the token in your environment, for example:
```bash
export HUGGINGFACEHUB_API_TOKEN=your_token
```

### 1.2 Gemini API (Google Generative AI)

**Purpose:**  
Generates research-oriented answers based on provided context, enhancing the depth and quality of responses.

**Usage:**  
The Gemini model is accessed using the `google.generativeai` library. For example:
```python
import google.generativeai as genai

genai.configure(api_key="YOUR_GOOGLE_API_KEY")
gemini_model = genai.GenerativeModel("gemini-pro")
prompt = "Your prompt here..."
response = gemini_model.generate_content(prompt)
```

**API Key Setup:**  
To use the Gemini API:
1. Obtain a Google Cloud account.
2. Enable the Generative AI API in your Google Cloud Console.
3. Create an API key.
4. Set the API key in your environment:
```bash
export GOOGLE_API_KEY=your_google_api_key
```
Alternatively, set the key directly in your code (not recommended for production).

---

## 2. Internal Endpoints and Functions

### 2.1 PDF Processing and FAISS Indexing

**Function:**  
`process_pdf(pdf_path)`

**Description:**  
Extracts text, tables, and images from an uploaded PDF, and builds a FAISS index for semantic search.

**Input:**  
`pdf_path` (string representing the file path of the PDF)

**Output:**  
Saves a FAISS index and associated text chunks to disk.

**Function:**  
`build_faiss_index(text_chunks, pdf_path)`

**Description:**  
Encodes text chunks using a SentenceTransformer and builds a FAISS index, saving both the index and text chunks.

**Input:**  
A list of text chunks and the PDF path.

**Output:**  
A FAISS index, embeddings, and text chunks are saved for later retrieval.

### 2.2 FAISS Search

**Function:**  
`search_faiss(query, faiss_index, embedding_model, chunks)`

**Description:**  
Searches the FAISS index by encoding the user query and retrieving the most relevant text chunks.

**Input:**  
- `query`: A string representing the user's query.
- `faiss_index`: The pre-built FAISS index.
- `embedding_model`: The model used for encoding text.
- `chunks`: The list of text chunks from the PDF.

**Output:**  
Returns a list of relevant text chunks.

---

## 3. Additional Notes

### 3.1 Caching and Performance

**Caching:**  
Embeddings and FAISS indexes are cached to avoid redundant computations and improve response time.

### 3.2 Versioning and Maintenance

**Versioning:**  
Regularly check the API documentation for Hugging Face and Gemini to ensure compatibility and update your integration if necessary.

**Maintenance:**  
Maintain a changelog to track updates and bug fixes.

### 3.3 Environment Variables

**Required Variables:**  
Ensure the following environment variables are set:

- `HUGGINGFACEHUB_API_TOKEN`: Your Hugging Face API token.
- `GOOGLE_API_KEY`: Your Google Generative AI API key.

---

This API reference provides a comprehensive overview of the APIs used within ArxivLensAI, including how to configure and obtain necessary API keys for external services, and describes the internal endpoints and functions that drive the application.
