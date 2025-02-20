import fitz  # PyMuPDF for text & image extraction
import pdfplumber  # For structured table extraction
import re
import os


def extract_text_from_pdf(pdf_path):
    """Extracts clean text and tables from a given PDF file."""
    doc = fitz.open(pdf_path)  # Open the PDF
    extracted_text = ""

    for page_num, page in enumerate(doc):
        # ✅ Extract plain text
        page_text = page.get_text("text") or ""  

        # ✅ Extract tables (using pdfplumber)
        table_text = extract_tables_from_pdf(pdf_path, page_num) or ""

        # ✅ Concatenate extracted text
        extracted_text += f"{page_text}\n{table_text}\n"

    return clean_text(extracted_text)

def extract_tables_from_pdf(pdf_path, page_number):
    """Extracts tables from a specific page using pdfplumber."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        if page_number < len(pdf.pages):
            table_page = pdf.pages[page_number]
            extracted_tables = table_page.extract_table()
            if extracted_tables:
                for row in extracted_tables:
                    tables.append(" | ".join(str(cell) if cell else "" for cell in row))

    return "\n".join(tables) if tables else ""  # ✅ Convert tables to a readable format


def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """Extracts images from a given PDF file and saves them as separate files."""
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

    return image_paths  # ✅ Return list of extracted image paths

def clean_text(text):
    """Cleans extracted text by removing extra spaces and fixing line breaks."""
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = re.sub(r'-\s', '', text)   # Fix hyphenated words
    return text.strip()
