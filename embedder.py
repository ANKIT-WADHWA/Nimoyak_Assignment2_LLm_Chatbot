import os

# Force CPU usage and disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def get_hf_embeddings():
    # Explicitly tell SentenceTransformer to use CPU device
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

def create_vector_store(chunks, save_path="faiss_index"):
    embeddings = get_hf_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print("✅ FAISS index created and saved.")

def load_vector_store(save_path="faiss_index"):
    embeddings = get_hf_embeddings()
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
