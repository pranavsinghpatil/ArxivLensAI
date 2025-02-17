import streamlit as st
from qa_system import generate_answer_huggingface  # Assuming you already have this function
from vector_store import search_faiss  # Same assumption here
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the FAISS index (or whatever mechanism you are using for retrieval)
faiss_index = None  # Load FAISS index here

# Streamlit Interface
st.title("AI-based Question Answering System")

query = st.text_input("Ask a question:")
if st.button("Submit"):
    if query:
        # Retrieve relevant chunks from the FAISS index
        retrieved_chunks = search_faiss(faiss_index, query, model, chunks=None, top_k=3)

        # Generate an answer using the retrieved chunks
        answer = generate_answer_huggingface(query, retrieved_chunks, model, tokenizer)
        
        # Display the retrieved chunks and the generated answer
        st.write("Retrieved Chunks:")
        for chunk in retrieved_chunks:
            st.write(f"- {chunk}")
        st.write("Generated Answer:")
        st.write(answer)
    else:
        st.write("Please enter a query!")
