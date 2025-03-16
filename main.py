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
    chunks_filename = os.path.join(faiss_indexes_dir, f"chunks_{faiss_index_filename}.pkl")
    tables_file = os.path.join(faiss_indexes_dir, f"tables_{faiss_index_filename}.md")
    image_texts_file = os.path.join(faiss_indexes_dir, f"image_texts_{faiss_index_filename}.txt")

    # Clean up existing files if force_reprocess
    if force_reprocess:
        print("[DEBUG] Force reprocessing - cleaning up existing files")
        for file in [faiss_index_path, chunks_filename, tables_file, image_texts_file]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"[DEBUG] Removed {file}")
                except Exception as e:
                    print(f"[WARNING] Could not remove {file}: {e}")

    # Extract text from PDF
    print("[DEBUG] Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text or not isinstance(text, str) or not text.strip():
        print("[ERROR] No valid text extracted from PDF")
        return
        
    print(f"[DEBUG] Extracted {len(text)} characters")
    
    # Split text into meaningful chunks
    text_chunks = []
    current_chunk = []
    current_length = 0
    
    # Split by sentences but keep context
    sentences = text.split(". ")
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add period back if it was removed by split
        if not sentence.endswith('.'):
            sentence += '.'
            
        # If adding this sentence would exceed chunk size, save current chunk
        if current_length + len(sentence) > 1000:  # Adjust chunk size as needed
            if current_chunk:
                text_chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    # Add final chunk
    if current_chunk:
        text_chunks.append(" ".join(current_chunk))
    
    print(f"[DEBUG] Created {len(text_chunks)} chunks")
    
    # Print first few chunks for verification
    for i, chunk in enumerate(text_chunks[:3]):
        print(f"[DEBUG] Chunk {i+1} preview: {chunk[:100]}...")

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
        if faiss_index is None or embeddings is None or chunks is None:
            print("[ERROR] Failed to build FAISS index")
            return
            
        print("[DEBUG] Saving chunks...")
        with open(chunks_filename, "wb") as f:
            pickle.dump(chunks, f)
            
        print("[DEBUG] Saving FAISS index...")
        faiss.write_index(faiss_index, faiss_index_path)
        
        print(f"[SUCCESS] Processing complete:")
        print(f"[SUCCESS] - Chunks saved: {chunks_filename}")
        print(f"[SUCCESS] - Tables: {tables_file}")
        print(f"[SUCCESS] - Images: {extracted_images_dir}")
        print(f"[SUCCESS] - Image texts: {image_texts_file}")
        print(f"[SUCCESS] - FAISS index: {faiss_index_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to build or save FAISS index: {e}")
        # Clean up any partially created files
        for file in [faiss_index_path, chunks_filename, tables_file, image_texts_file]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as cleanup_error:
                    print(f"[WARNING] Could not clean up {file}: {cleanup_error}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Please provide a PDF file path.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_pdf(pdf_path)
