import streamlit as st
import os

# Force CPU and avoid parallel tokenizer errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loader import load_and_chunk_documents
from embedder import create_vector_store, load_vector_store
from query_engine import get_answer

st.title("Document Question Answering App")

# Step 1: Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF documents", type=["pdf"], accept_multiple_files=True
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")

    st.info("Creating FAISS index from uploaded documents...")
    chunks = load_and_chunk_documents(DATA_DIR)
    create_vector_store(chunks)
    st.success("FAISS index created!")

if os.path.exists("faiss_index"):
    vectorstore = load_vector_store()
    query = st.text_input("Ask me anything based on the uploaded documents:")
    if query:
        with st.spinner("Generating answer..."):
            answer = get_answer(vectorstore, query)
        st.markdown(f"**Answer:** {answer}")
else:
    st.warning("Please upload PDF documents to build the search index first.")
