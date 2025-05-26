from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def create_vector_store(chunks, save_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print("✅ FAISS index created and saved.")

def load_vector_store(save_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
