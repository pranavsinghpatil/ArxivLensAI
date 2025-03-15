import faiss
import pickle
import os
import sys
from extract_text import extract_text_from_pdf, extract_tables_from_pdf, extract_images_from_pdf, extract_text_from_images
from utils import get_faiss_index_filename, get_chunks_filename
from vector_store import build_faiss_index

# Directory setup
project_dir = os.path.dirname(os.path.abspath(__file__))
faiss_indexes_dir = os.path.join(project_dir, "faiss_indexes")
extracted_images_dir = os.path.join(project_dir, "extracted_images")
os.makedirs(faiss_indexes_dir, exist_ok=True)
os.makedirs(extracted_images_dir, exist_ok=True)

def process_pdf(pdf_path, force_reprocess=False):
    """Processes a PDF to extract text, tables, images, and builds a FAISS index."""
    print(f"[DEBUG] Starting to process PDF: {pdf_path}")

    faiss_index_filename = get_faiss_index_filename(pdf_path)
    faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)
    tables_file = os.path.join(faiss_indexes_dir, f"tables_{faiss_index_filename}.md")
    image_texts_file = os.path.join(faiss_indexes_dir, f"image_texts_{faiss_index_filename}.txt")

    if not force_reprocess and os.path.exists(faiss_index_path):
        print(f"[INFO] FAISS index already exists: {faiss_index_path}. Skipping index building.")
        return

    print("[DEBUG] Calling extract_text_from_pdf...")
    text = extract_text_from_pdf(pdf_path)
    
    if not text or not isinstance(text, str):
        print(f"[ERROR] Invalid text extracted from PDF: type={type(text)}, content={text!r}")
        return
        
    if not text.strip():
        print("[ERROR] No valid text extracted from the PDF.")
        return
        
    print(f"[DEBUG] Successfully extracted {len(text)} characters of text")
    text_chunks = [chunk for chunk in text.split(". ") if chunk.strip()]
    print(f"[DEBUG] Created {len(text_chunks)} text chunks")

    print("[DEBUG] Extracting tables...")
    with open(tables_file, "w", encoding='utf-8') as f:
        for page_number in range(10):  # Extract tables from first 10 pages
            table_text = extract_tables_from_pdf(pdf_path, page_number)
            if table_text:
                f.write(f"\n## Page {page_number + 1} Tables:\n{table_text}\n")

    print("[DEBUG] Extracting images...")
    image_paths = extract_images_from_pdf(pdf_path, extracted_images_dir)
    print(f"[DEBUG] Extracted {len(image_paths)} images")

    # Extract text from images and save to a file
    print("[DEBUG] Extracting text from images...")
    image_texts = extract_text_from_images(image_paths)
    with open(image_texts_file, "w", encoding='utf-8') as f:
        f.write("\n".join(text for text in image_texts if text))

    print("[DEBUG] Building FAISS index...")
    try:
        faiss_index, embeddings, chunks = build_faiss_index(text_chunks, pdf_path)
        
        print("[DEBUG] Saving chunks...")
        chunks_file = os.path.join(faiss_indexes_dir, f"chunks_{faiss_index_filename}.pkl")
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)

        print("[DEBUG] Saving FAISS index...")
        faiss.write_index(faiss_index, faiss_index_path)

        print(f"[SUCCESS] Processing complete for {pdf_path}")
        print(f"[SUCCESS] - FAISS index: {faiss_index_path}")
        print(f"[SUCCESS] - Tables: {tables_file}")
        print(f"[SUCCESS] - Images: {extracted_images_dir}")
        print(f"[SUCCESS] - Image texts: {image_texts_file}")
        
    except Exception as e:
        print(f"[ERROR] Failed to build or save FAISS index: {e}")
        # Clean up any partially created files
        for file in [faiss_index_path, tables_file, image_texts_file]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as cleanup_error:
                    print(f"[WARNING] Failed to clean up {file}: {cleanup_error}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Please provide a PDF file path.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_pdf(pdf_path)
