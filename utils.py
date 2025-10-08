import hashlib
import os
import streamlit as st

# Safely resolve API keys from environment first, then Streamlit secrets only if a secrets.toml exists
def _get_secret_or_env(secret_key, env_key):
    env_value = os.getenv(env_key, "").strip()
    if env_value:
        return env_value
    # Detect secrets.toml presence to avoid triggering Streamlit warnings locally
    try:
        cwd = os.getcwd()
        project_secrets = os.path.join(cwd, ".streamlit", "secrets.toml")
        home_dir = os.path.expanduser("~")
        home_secrets = os.path.join(home_dir, ".streamlit", "secrets.toml")
        if os.path.exists(project_secrets) or os.path.exists(home_secrets):
            # Only touch st.secrets if a file actually exists
            if hasattr(st, "secrets") and secret_key in st.secrets:
                value = st.secrets[secret_key]
                if isinstance(value, str) and value.strip():
                    return value.strip()
    except Exception:
        pass
    return ""

GOOGLE_API_KEY = _get_secret_or_env("GOOGLE_API_KEY", "GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = _get_secret_or_env("HUGGINGFACE_API_KEY", "HUGGINGFACE_API_KEY")

def get_chunks_filename(pdf_path):
    """Generates a unique filename for text chunks based on the PDF path."""
    pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    return f"chunks_{pdf_hash}.pkl"

def get_faiss_index_filename(pdf_path):
    """Generates a unique filename for FAISS index based on the PDF path."""
    if pdf_path is None:
        raise ValueError("pdf_path is None. Ensure a PDF is uploaded before generating FAISS index filename.")

    pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    return f"faiss_index_{pdf_hash}.index"

def expand_query(query, memory, max_expansions=3):
    """
    Expands the query using related past queries from chat history.
    """
    expanded_terms = set(query.lower().split())

    # Retrieve related queries from memory
    past_queries = [m["content"].lower() for m in memory if m["role"] == "user"]
    for past_query in past_queries:
        if any(word in past_query for word in expanded_terms):
            expanded_terms.update(past_query.split())
        if len(expanded_terms) >= max_expansions:
            break

    return " ".join(expanded_terms)

# Keywords for full context queries
full_context_keywords = [
    "summary", "summarize", "overview", "key points", "high-level explanation",
    "explain all", "explain fully", "detailed explanation", "comprehensive analysis",
    "break down", "step-by-step", "go in-depth", "expand on",
    "simplest", "explain simply", "easy explanation", "beginner-friendly",
    "explain like I'm five (ELI5)", "make it intuitive", "basic version",
    "compare", "contrast", "differences", "similarities", "how does it differ from",
    "real-world example", "practical application", "use cases", "industry examples",
    "where is this used", "applied research",
    "limitations", "weaknesses", "challenges", "trade-offs", "bottlenecks",
    "what are the drawbacks", "how can this be improved"
]
