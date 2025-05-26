import os
from dotenv import load_dotenv
load_dotenv()

# Force CPU usage and tokenizer parallelism setting
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loader import load_and_chunk_documents
from embedder import create_vector_store, load_vector_store
from query_engine import get_answer_from_query

def main():
    if not os.path.exists("faiss_index"):
        print("📄 No FAISS index found. Creating index from documents...")
        chunks = load_and_chunk_documents("data")
        create_vector_store(chunks)

    vectorstore = load_vector_store()

    print("Ask me anything based on the uploaded documents.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = get_answer_from_query(vectorstore, query)
        print("Bot:", answer)

if __name__ == "__main__":
    main()
