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
            if text:  # Ensure text is not None or empty
                extracted_texts.append(text)
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
    return extracted_texts

def extract_tables_from_pdf(pdf_path, page_number):
    """Extracts tables from a specific page using pdfplumber and parses them into structured data."""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < len(pdf.pages):
                table_page = pdf.pages[page_number]
                extracted_tables = table_page.extract_table()
                if extracted_tables:
                    for row in extracted_tables:
                        tables.append(" | ".join(str(cell) if cell else "" for cell in row))
    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")

    return "\n".join(tables) if tables else ""  # Convert tables to a readable format

def extract_text_from_pdf(pdf_path):
    """Extracts clean text and tables from a given PDF file."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return ""
        
    try:
        doc = fitz.open(pdf_path)  # Open the PDF
        if doc is None:
            print(f"Error: Could not open PDF {pdf_path}")
            return ""
            
        extracted_text = []

        for page_num, page in enumerate(doc):
            # Extract plain text
            try:
                page_text = page.get_text("text") or ""  # Use empty string if None
                extracted_text.append(page_text)
                
                # Extract tables (using pdfplumber)
                table_text = extract_tables_from_pdf(pdf_path, page_num) or ""
                if table_text:
                    extracted_text.append(table_text)
            except Exception as e:
                print(f"Warning: Error extracting text from page {page_num + 1}: {e}")
                continue

        doc.close()  # Properly close the document
        final_text = "\n".join(text for text in extracted_text if text)
        return clean_text(final_text) if final_text else ""
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return ""

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
    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                image = doc.extract_image(xref)
                image_bytes = image["image"]
                img_filename = f"{output_folder}/pdf_page_{page_num}_img_{img_index}.png"

                with open(img_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                    image_paths.append(img_filename)
    except Exception as e:
        print(f"Error extracting images from {pdf_path}: {e}")

    return image_paths  # Return list of extracted image paths

def clean_text(text):
    """Cleans extracted text by removing extra spaces and fixing line breaks."""
    if not text or not isinstance(text, str):
        print(f"Warning: Invalid text received in clean_text: {type(text)}")
        return ""
        
    try:
        text = text.encode("utf-8", "ignore").decode("utf-8")
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
        text = re.sub(r'-\s', '', text)   # Fix hyphenated words
        return text.strip()
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""
