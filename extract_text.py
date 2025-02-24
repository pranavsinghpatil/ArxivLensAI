import fitz  # PyMuPDF for text & image extraction
import pdfplumber  # For structured table extraction
import re
import os
import pytesseract
from PIL import Image

def extract_text_from_images(image_paths):
    """Extracts text from images using OCR."""
    extracted_texts = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            extracted_texts.append(text)
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
    return extracted_texts

def extract_tables_from_pdf(pdf_path, page_number):
    """Extracts tables from a specific page using pdfplumber and parses them into structured data."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        if page_number < len(pdf.pages):
            table_page = pdf.pages[page_number]
            extracted_tables = table_page.extract_table()
            if extracted_tables:
                for row in extracted_tables:
                    tables.append(" | ".join(str(cell) if cell else "" for cell in row))

    return "\n".join(tables) if tables else ""  # Convert tables to a readable format

def extract_text_from_pdf(pdf_path):
    """Extracts clean text and tables from a given PDF file."""
    doc = fitz.open(pdf_path)  # Open the PDF
    extracted_text = ""

    for page_num, page in enumerate(doc):
        # Extract plain text
        page_text = page.get_text("text") or ""

        # Extract tables (using pdfplumber)
        table_text = extract_tables_from_pdf(pdf_path, page_num) or ""

        # Concatenate extracted text
        extracted_text += f"{page_text}\n{table_text}\n"

    return clean_text(extracted_text)

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """
    Extracts images from a given PDF file and saves them as separate files.

    Args:
        pdf_path (str): The file path to the PDF.
        output_folder (str): The directory where extracted images will be saved.

    Returns:
        list: A list of file paths to the extracted images.
    """
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            image = doc.extract_image(xref)
            image_bytes = image["image"]
            img_filename = f"{output_folder}/pdf_page_{page_num}_img_{img_index}.png"

            with open(img_filename, "wb") as img_file:
                img_file.write(image_bytes)
                image_paths.append(img_filename)

    return image_paths  # Return list of extracted image paths

def clean_text(text):
    """Cleans extracted text by removing extra spaces and fixing line breaks."""
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = re.sub(r'-\s', '', text)   # Fix hyphenated words
    return text.strip()
