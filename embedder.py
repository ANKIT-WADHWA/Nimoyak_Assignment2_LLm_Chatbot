import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

class CpuSafeHuggingFaceEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Initialize model explicitly on CPU
        self.model = SentenceTransformer(model_name, device='cpu')

    def embed_documents(self, texts):
        # Encode list of documents (texts)
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        # Encode single query text
        return self.model.encode(text, convert_to_tensor=False).tolist()

def get_hf_embeddings():
    return CpuSafeHuggingFaceEmbeddings()

def create_vector_store(chunks, save_path="faiss_index"):
    embeddings = get_hf_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print("✅ FAISS index created and saved.")

def load_vector_store(save_path="faiss_index"):
    embeddings = get_hf_embeddings()
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
