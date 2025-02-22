# utils.py

# import faiss
# import pickle
# import os
import hashlib
# import sys
import nltk
from nltk.corpus import wordnet

nltk.download("wordnet")

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
    Expands the query using:
    1. Synonyms from WordNet
    2. Related past queries from chat history
    """
    expanded_terms = set(query.lower().split())

    # ✅ 1. Add synonyms from WordNet
    for word in query.split():
        synonyms = wordnet.synsets(word)
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_terms.add(lemma.name().replace("_", " "))
                if len(expanded_terms) >= max_expansions:
                    break
            if len(expanded_terms) >= max_expansions:
                break

    # ✅ 2. Retrieve related queries from memory
    past_queries = [m["content"].lower() for m in memory if m["role"] == "user"]
    for past_query in past_queries:
        if any(word in past_query for word in expanded_terms):
            expanded_terms.update(past_query.split())
        if len(expanded_terms) >= max_expansions:
            break

    return " ".join(expanded_terms)

full_context_keywords = [
    # ✅ Summarization Requests
    "summary", "summarize", "overview", "key points", "high-level explanation",

    # ✅ Deep Explanations & Conceptual Understanding
    "explain all", "explain fully", "detailed explanation", "comprehensive analysis",
    "break down", "step-by-step", "go in-depth", "expand on",

    # ✅ Mathematical & Technical Explanations
    "explain the math", "derivation", "proof", "formulation", "stepwise solution",
    "mathematical intuition", "equation breakdown", "explain formula",

    # ✅ Simplifications & Layman Explanations
    "simplest", "explain simply", "easy explanation", "beginner-friendly",
    "explain like I'm five (ELI5)", "make it intuitive", "basic version",

    # ✅ Comparisons & Contrasts
    "compare", "contrast", "differences", "similarities", "how does it differ from",

    # ✅ Use Cases & Applications
    "real-world example", "practical application", "use cases", "industry examples",
    "where is this used", "applied research",

    # ✅ Critical Analysis & Limitations
    "limitations", "weaknesses", "challenges", "trade-offs", "bottlenecks",
    "what are the drawbacks", "how can this be improved"
]
