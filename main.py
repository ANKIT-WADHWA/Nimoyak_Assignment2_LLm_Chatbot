from loader import load_and_chunk_documents
from embedder import create_vector_store, load_vector_store
from query_engine import get_answer
import os

def main():
    if not os.path.exists("faiss_index"):
        print("📄 No FAISS index found. Creating index from documents...")
        chunks = load_and_chunk_documents("data")
        create_vector_store(chunks)

    vectorstore = load_vector_store()

    print("🧠 Ask me anything based on the uploaded documents.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = get_answer(vectorstore, query)
        print("Bot:", answer)

if __name__ == "__main__":
    main()
