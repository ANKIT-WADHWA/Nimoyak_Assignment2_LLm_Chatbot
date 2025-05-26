from dotenv import load_dotenv
import os
from langchain_groq.chat_models import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()  

def get_answer(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print(" Using Groq's LLaMA 3 model via API")
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt_template = """
You are an assistant answering questions based only on the given context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    prompt = PromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query": query})
    return result["result"]
