# utils.py
import faiss
import pickle
import os
import hashlib
import sys

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

# styles.py

CUSTOM_STYLES = """
    /* Base Chat Container */
    .chat-container {
        display: flex;
        flex-direction: column-reverse;
        padding: 20px 20px 80px 20px;
        height: calc(100vh - 150px);
        overflow-y: auto;
        background-color: #f5f5f5;
    }

    /* User Message (Right-aligned) */
    .user-message {
        background-color: #2b7fff;
        color: white;
        padding: 12px 16px;
        border-radius: 15px 15px 0 15px;
        max-width: 70%;
        margin: 8px 0;
        align-self: flex-end;
        word-wrap: break-word;
    }

    /* Bot Message (Left-aligned, no box) */
    .bot-message {
        color: #333;
        padding: 12px 16px;
        max-width: 75%;
        margin: 8px 0;
        align-self: flex-start;
        word-wrap: break-word;
    }

    /* Fixed Input Container */
    .chat-input-wrapper {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 16px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }

    /* Input Box with Send Button */
    .chat-input-container {
        display: flex;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
        background: #f5f5f5;
        border-radius: 24px;
        padding: 4px 4px 4px 16px;
        position: relative;
    }

    /* Text Input Styling */
    .chat-input {
        flex: 1;
        border: none;
        background: transparent;
        padding: 12px;
        font-size: 16px;
        outline: none;
    }

    /* Send Button Styling */
    .send-button {
        background: #2b7fff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background-color 0.2s;
        margin-left: 8px;
    }

    .send-button:hover {
        background: #1a6eeb;
    }

    /* Hide Streamlit Footer */
    footer {
        display: none !important;
    }

    /* Minimize Sidebar */
    .st-emotion-cache-1l269bu {
        max-width: 140px !important;
        min-width: 120px !important;
    }
"""

PAGE_STYLES = """
    <style>
        /* Fix Sidebar Width */
        .st-emotion-cache-1l269bu {  
            min-width: 120px !important;  
            max-width: 140px !important;  
        }

        /* Remove Bottom White Strip */
        footer {visibility: hidden;}
    </style>
"""

EXTERNAL_DEPENDENCIES = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
"""