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
    print(f"[DEBUG] Starting PDF extraction from: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found at {pdf_path}")
        return ""
        
    try:
        print("[DEBUG] Opening PDF with PyMuPDF...")
        doc = fitz.open(pdf_path)  # Open the PDF
        if doc is None:
            print(f"[ERROR] Could not open PDF {pdf_path}")
            return ""
            
        extracted_text = []
        print(f"[DEBUG] PDF has {len(doc)} pages")

        for page_num, page in enumerate(doc):
            print(f"[DEBUG] Processing page {page_num + 1}")
            # Extract plain text
            try:
                page_text = page.get_text("text")
                print(f"[DEBUG] Page {page_num + 1} text type: {type(page_text)}, length: {len(page_text) if page_text else 0}")
                
                if page_text:
                    extracted_text.append(page_text)
                    
                # Extract tables (using pdfplumber)
                table_text = extract_tables_from_pdf(pdf_path, page_num)
                if table_text:
                    print(f"[DEBUG] Found table text on page {page_num + 1}")
                    extracted_text.append(table_text)
            except Exception as e:
                print(f"[ERROR] Failed extracting text from page {page_num + 1}: {e}")
                continue

        doc.close()
        print("[DEBUG] PDF document closed")
        
        # Join all extracted text
        final_text = "\n".join(text for text in extracted_text if text and isinstance(text, str))
        print(f"[DEBUG] Final text length before cleaning: {len(final_text)}")
        
        if not final_text:
            print("[WARNING] No text was extracted from the PDF")
            return ""
            
        cleaned_text = clean_text(final_text)
        print(f"[DEBUG] Text length after cleaning: {len(cleaned_text)}")
        return cleaned_text
        
    except Exception as e:
        print(f"[ERROR] Failed processing PDF {pdf_path}: {e}")
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
    print(f"[DEBUG] clean_text received text of type: {type(text)}")
    
    if text is None:
        print("[ERROR] clean_text received None")
        return ""
        
    if not isinstance(text, str):
        print(f"[ERROR] clean_text received non-string type: {type(text)}")
        return ""
    
    if not text.strip():
        print("[WARNING] clean_text received empty or whitespace-only string")
        return ""
        
    try:
        print(f"[DEBUG] Original text length: {len(text)}")
        # Remove any null bytes that might cause encoding issues
        text = text.replace('\x00', '')
        
        # Handle encoding
        text = text.encode("utf-8", "ignore").decode("utf-8")
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'-\s', '', text)
        text = text.strip()
        
        print(f"[DEBUG] Cleaned text length: {len(text)}")
        return text
    except Exception as e:
        print(f"[ERROR] Failed to clean text: {e}")
        print(f"[DEBUG] Text that caused error: {repr(text)[:100]}...")
        return ""
