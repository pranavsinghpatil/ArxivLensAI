import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    """Extracts clean text from a given PDF file."""
    doc = fitz.open(pdf_path)  # Open the PDF
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"  # Extract text from each page

    return text.strip()  # Remove extra whitespace

def clean_text(text):
    """Cleans extracted text by removing extra spaces and fixing line breaks."""
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = re.sub(r'-\s', '', text)   # Fix hyphenated words
    return text.strip()
