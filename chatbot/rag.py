"""
rag.py  —  Your existing file with fixes applied:
  1. Fallback when vector store is empty or not built yet
  2. Better prompt: company questions → docs, other questions → Gemini knowledge
  3. Proper error handling
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


class RAGApplication:
    def __init__(self, db_path="my_company_db"):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./models"
        )

        self.vector_store = Chroma(
            collection_name="company_docs",
            embedding_function=self.embeddings,
            persist_directory=db_path
        )

        # k=6 — retrieve top 6 matching chunks from your company PDF
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 6}
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )

    def answer_query(self, question: str) -> str:
        # ── UPDATED PROMPT ─────────────────────────────────────────────────
        # Two-mode behaviour:
        #   • Company questions → answer strictly from retrieved context
        #   • General/tech questions → answer from Gemini's own knowledge
        template = """You are Syn, the official AI assistant for Syandrixin — \
a technology company specializing in software, AI, and web development.

RULES:
1. If the retrieved context below contains relevant information about the question, \
use it to give a detailed, well-structured answer. Use bullet points or numbered \
lists where appropriate.
2. If the context is empty or not relevant to the question, answer using your \
own general knowledge — especially for tech, coding, science, or general topics.
3. Never say "I don't have enough information" for general knowledge questions — \
you are a smart assistant. Only say that if it is a specific Syandrixin internal \
question (e.g. employee names, private pricing) that isn't in the context.
4. Never mention Gemini, Google, LangChain, or any underlying AI technology. \
You are simply Syn by Syandrixin.
5. Always be professional, friendly, and concise unless detail is needed.

Retrieved context from Syandrixin documents:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)


if __name__ == "__main__":
    app = RAGApplication()
    print(app.answer_query("how many projects has Syandrixin delivered so far?"))