# app.py
from sentence_transformers import SentenceTransformer
import streamlit as st
import pickle
from qa_system import generate_answer_huggingface, set_api_keys
from vector_store import search_faiss, load_faiss_index
import os
import faiss
from main import process_pdf
from extract_text import extract_text_from_images
from utils import get_faiss_index_filename, get_chunks_filename, full_context_keywords,  GOOGLE_API_KEY, HUGGINGFACE_API_KEY
import pandas as pd
from fuzzywuzzy import process
import time

# ✅ Fix GRPC error
os.environ["GRPC_DNS_RESOLVER"] = "ares"

# ✅ Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=HUGGINGFACE_API_KEY) if HUGGINGFACE_API_KEY else None

# ✅ Set up directories
project_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(project_dir, "temp")
faiss_indexes_dir = os.path.join(project_dir, "faiss_indexes")
extracted_images_dir = os.path.join(project_dir, "extracted_images")
tables_dir = os.path.join(project_dir, "extracted_tables")
static_dir = os.path.join(project_dir, "static")
# ✅ Ensure directories exist
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(faiss_indexes_dir, exist_ok=True)
os.makedirs(extracted_images_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

# ✅ Default Research Paper
default_paper_path = os.path.join(temp_dir, "Attention Is All You Need(default_research_paper).pdf")

# ✅ Streamlit UI setup
st.set_page_config(page_title="📝 ArxivLensAI", layout="wide")

if "memory" not in st.session_state:
    st.session_state.memory = []

# ✅ Display Fixed Header
st.markdown("<h3 style='text-align: center;'>🤖 ArxivLensAI (An AI-Powered Research Assistant)</h3>", unsafe_allow_html=True)

# 📌 Sidebar - Multiple PDF Upload
st.sidebar.header("📄 Upload Research Papers")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# ✅ Store uploaded papers
available_papers = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        print(f"[DEBUG] Processing uploaded file: {uploaded_file.name}")
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ✅ Show notification instead of sidebar clutter
        st.toast(f"✅ {uploaded_file.name} uploaded!", icon="📄")

        # ✅ Process PDF if necessary
        try:
            faiss_index_filename = get_faiss_index_filename(pdf_path)
            faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)
            chunks_filename = os.path.join(faiss_indexes_dir, f"chunks_{faiss_index_filename}.pkl")

            print(f"[DEBUG] Checking for existing index at: {faiss_index_path}")
            if not os.path.exists(faiss_index_path) or not os.path.exists(chunks_filename):
                print("[DEBUG] Index not found, processing PDF...")
                process_pdf(pdf_path)
            else:
                print("[DEBUG] Found existing index")

            available_papers[uploaded_file.name] = pdf_path
        except Exception as e:
            print(f"[ERROR] Failed to process {uploaded_file.name}: {e}")
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

# ✅ Load Default Paper if No Uploads
if not available_papers:
    st.sidebar.info("📌 Using default paper: Attention Is All You Need")
    if os.path.exists(default_paper_path):
        print(f"[DEBUG] Loading default paper from: {default_paper_path}")
        available_papers["Attention Is All You Need (Default)"] = default_paper_path
        
        # Process default paper if needed
        try:
            faiss_index_filename = get_faiss_index_filename(default_paper_path)
            faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)
            chunks_filename = os.path.join(faiss_indexes_dir, f"chunks_{faiss_index_filename}.pkl")
            
            print(f"[DEBUG] Checking for default paper index at: {faiss_index_path}")
            if not os.path.exists(faiss_index_path) or not os.path.exists(chunks_filename):
                print("[DEBUG] Processing default paper...")
                process_pdf(default_paper_path)
            else:
                print("[DEBUG] Found existing index for default paper")
        except Exception as e:
            print(f"[ERROR] Failed to process default paper: {e}")
            st.error("⚠️ Error processing default paper!")
    else:
        st.error("⚠️ Default research paper is missing! Please upload a file.")

# ✅ Dropdown to Select Research Papers
selected_papers = st.sidebar.multiselect(
    "📂 Select Research Papers",
    list(available_papers.keys()),
    default=list(available_papers.keys())
)
st.session_state.selected_papers = [available_papers[p] for p in selected_papers]

# ------------------------------------------------------------

if not GOOGLE_API_KEY or not HUGGINGFACE_API_KEY:
    st.sidebar.markdown("---")
    st.sidebar.info("Enter both API keys to proceed.", icon='⬇️')

    # Google AI API key input
    gapi_key = st.sidebar.text_input(
        "🔑 Google AI API key",
        type="password",
        value=st.session_state.get("gapi_key", ""),
        help="Enter your Google AI API key. It will be securely stored.",
        key="gapi_key_input"
    )

    # Hugging Face API key input
    hapi_key = st.sidebar.text_input(
        "🔑 Hugging Face API key",
        type="password",
        value=st.session_state.get("hapi_key", ""),
        help="Enter your Hugging Face API key. It will be securely stored.",
        key="hapi_key_input"
    )

    # Store the API keys in session state
    if gapi_key:
        st.session_state["gapi_key"] = gapi_key

    if hapi_key:
        st.session_state["hapi_key"] = hapi_key

    # Optional: Validate the API keys
    if gapi_key and hapi_key and gapi_key.startswith("AIza") and hapi_key.startswith("hf_"):
        st.sidebar.success("Both API keys have been entered successfully!", icon='✅')
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", token=hapi_key)

    elif gapi_key and not gapi_key.startswith("AIza"):
        st.sidebar.warning("Please enter a valid Google AI API key!", icon="⚠️")
    elif gapi_key and gapi_key.startswith("AIza"):
        st.sidebar.info("Google AI API key entered. Waiting for Hugging Face API key.")

    elif hapi_key and not hapi_key.startswith("hf_"):
        st.sidebar.warning("Please enter a valid Hugging Face API key!", icon="⚠️")
    elif hapi_key and hapi_key.startswith("hf_"):
        st.sidebar.info("Hugging Face API key entered. Waiting for Google AI API key.")

    # Set API keys in qa_system
    set_api_keys(gapi_key, hapi_key)
else:
    # st.sidebar.markdown("---")
    # st.sidebar.success('API key already provided!', icon='✅')
    set_api_keys(gapi_key=GOOGLE_API_KEY, hapi_key=HUGGINGFACE_API_KEY)
st.sidebar.markdown("---")
st.sidebar.markdown("### ArxivLensAI")
st.sidebar.markdown("Version: `1.0.0`")
st.sidebar.markdown("Author: [PranavSingh Patil]('url')")
st.sidebar.markdown("[Report a bug]('https://github.com/pranavsinghpatil/ArxivLensAI/issues')")
st.sidebar.markdown("[GitHub repo]('https://github.com/pranavsinghpatil/ArxivLensAI')")
st.sidebar.markdown("License: [MIT]('https://github.com/pranavsinghpatil/ArxivLensAI/blob/main/LICENSE')")

# ------------------------------------------------------------

# ✅ Ensure Selected Papers are Processed
for pdf_path in st.session_state.selected_papers:
    try:
        faiss_index_filename = get_faiss_index_filename(pdf_path)
        faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)

        if not os.path.exists(faiss_index_path):
            process_pdf(pdf_path)

    except Exception as e:
        st.error(f"❌ Error processing {pdf_path}: {str(e)}")

# ✅ Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Chatbot UI - Display Messages
chat_container = st.container()

# ✅ Chat Input (Fixed at Bottom)
query = st.chat_input("Ask your question here...")

# Initialize conversation history if not already present
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Display conversation history
for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.chat_message("user", avatar=os.path.join(static_dir, "icons", "user-icon.png")).markdown(message["content"])
    else:
        st.chat_message("assistant", avatar=os.path.join(static_dir, "icons", "bot-icon.png")).markdown(message["content"])

def is_full_context_query(query, keywords, threshold=80):
    """Detects if the query is asking for full context using fuzzy matching."""
    matches = process.extract(query.lower(), keywords, limit=len(keywords))
    best_match = max(matches, key=lambda x: x[1])
    return best_match[1] >= threshold

# Handle user query
if query:
    print(f"\n[DEBUG] Processing new query: {query}")
    
    # Display user message
    st.chat_message("user", avatar=os.path.join(static_dir, "icons", "user-icon.png")).markdown(query)
    st.session_state.conversation_history.append({"role": "user", "content": query})

    # Display assistant message with spinner while processing
    with st.chat_message("assistant", avatar=os.path.join(static_dir, "icons", "bot-icon.png")):
        with st.spinner("Thinking..."):
            try:
                # Load FAISS indexes for selected papers
                print(f"[DEBUG] Selected papers: {st.session_state.selected_papers}")
                all_chunks = []
                all_indexes = []
                
                for pdf_path in st.session_state.selected_papers:
                    try:
                        print(f"[DEBUG] Loading index for: {pdf_path}")
                        index, chunks = load_faiss_index(pdf_path)
                        if index is not None and chunks is not None:
                            all_indexes.append(index)
                            all_chunks.extend(chunks)
                            print(f"[DEBUG] Loaded {len(chunks)} chunks from {pdf_path}")
                        else:
                            print(f"[WARNING] Failed to load index for {pdf_path}")
                    except Exception as e:
                        print(f"[ERROR] Error loading index for {pdf_path}: {e}")
                        continue

                if not all_chunks or not all_indexes:
                    st.error("❌ No valid indexes found for the selected papers!")
                    st.stop()

                print(f"[DEBUG] Total chunks loaded: {len(all_chunks)}")
                
                # Search for relevant chunks
                relevant_chunks = []
                for index, chunks in zip(all_indexes, [all_chunks]):
                    try:
                        chunks_found = search_faiss(query, index, embedding_model, chunks, 
                                                  st.session_state.conversation_history)
                        if chunks_found and chunks_found[0] != "Error:":
                            relevant_chunks.extend(chunks_found)
                            print(f"[DEBUG] Found {len(chunks_found)} relevant chunks")
                    except Exception as e:
                        print(f"[ERROR] FAISS search failed: {e}")
                        continue

                if not relevant_chunks:
                    st.error("❌ Could not find relevant information in the papers!")
                    st.stop()

                # Generate answer
                answer = generate_answer_huggingface(
                    query=query,
                    retrieved_chunks=relevant_chunks,
                    memory=st.session_state.conversation_history,
                    image_texts=[],  # TODO: Add image text support
                    table_texts=[],   # TODO: Add table text support
                    full_context=False
                )

                st.markdown(answer)
                st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                print(f"[ERROR] Failed to process query: {e}")
                st.error(f"❌ Error: {str(e)}")
