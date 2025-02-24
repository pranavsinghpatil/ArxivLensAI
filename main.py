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

    faiss_index_filename = get_faiss_index_filename(pdf_path)
    faiss_index_path = os.path.join(faiss_indexes_dir, faiss_index_filename)
    tables_file = os.path.join(faiss_indexes_dir, f"tables_{faiss_index_filename}.md")
    image_texts_file = os.path.join(faiss_indexes_dir, f"image_texts_{faiss_index_filename}.txt")

    if not force_reprocess and os.path.exists(faiss_index_path):
        print(f"[INFO] FAISS index already exists: {faiss_index_path}. Skipping index building.")
        return

    print("[INFO] Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("[ERROR] No valid text extracted from the PDF.")
        return
    text_chunks = text.split(". ")

    print("[INFO] Extracting tables...")
    with open(tables_file, "w") as f:
        for page_number in range(10):  # Extract tables from first 10 pages
            table_text = extract_tables_from_pdf(pdf_path, page_number)
            if table_text:
                f.write(f"\n## Page {page_number + 1} Tables:\n{table_text}\n")

    print("[INFO] Extracting images...")
    image_paths = extract_images_from_pdf(pdf_path, extracted_images_dir)
    print(f"[INFO] Extracted {len(image_paths)} images.")

    # Extract text from images and save to a file
    image_texts = extract_text_from_images(image_paths)
    with open(image_texts_file, "w") as f:
        f.write("\n".join(image_texts))

    print("[INFO] Building FAISS index...")
    faiss_index, embeddings, chunks = build_faiss_index(text_chunks, pdf_path)

    with open(f"faiss_indexes/chunks_{faiss_index_filename}.pkl", "wb") as f:
        pickle.dump(chunks, f)

    faiss.write_index(faiss_index, faiss_index_path)

    print(f"[SUCCESS] FAISS index saved to {faiss_index_path}.")
    print(f"[SUCCESS] Extracted tables saved to {tables_file}.")
    print(f"[SUCCESS] Images saved to {extracted_images_dir}.")
    print(f"[SUCCESS] Image texts saved to {image_texts_file}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Please provide a PDF file path.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_pdf(pdf_path)
