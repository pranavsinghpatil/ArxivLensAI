import fitz  # PyMuPDF
import re


def extract_text_from_pdf(pdf_path):
    """Extracts clean text and tables from a given PDF file using PyMuPDF (fitz)."""
    doc = fitz.open(pdf_path)  # Open the PDF
    extracted_text = ""

    for page in doc:
        extracted_text += page.get_text("text") + "\n"  # Extract text

        # Extract tables using PyMuPDF (fitz)
        table_text = extract_tables_from_page(page)
        extracted_text += table_text + "\n"

    return clean_text(extracted_text)

def extract_tables_from_page(page):
    """Extracts tables from a given PDF page using PyMuPDF (fitz)."""
    tables = []
    words = page.get_text("words")  # Get all words with their coordinates

    # Group words into table-like structures
    if words:
        words.sort(key=lambda w: (w[1], w[0]))  # Sort by (y, then x)
        table = []
        last_y = None

        for w in words:
            x, y, x1, y1, text, _, _, _ = w
            if last_y is None or abs(y - last_y) < 10:  # Small Y difference → same row
                table.append(text)
            else:
                tables.append(" | ".join(table))
                table = [text]
            last_y = y

        if table:
            tables.append(" | ".join(table))

    return "\n".join(tables) if tables else ""

def clean_text(text):
    """Cleans extracted text by removing extra spaces and fixing line breaks."""
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = re.sub(r'-\s', '', text)   # Fix hyphenated words
    return text.strip()

