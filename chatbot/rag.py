"""
rag.py — Complete rewrite, path issue fixed
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# ── Absolute path so it works from any working directory ──────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "my_company_db")   # always project root


class RAGApplication:
    def __init__(self, db_path=None):
        db_path = db_path or DB_PATH

        print(f"[RAG] Loading vector store from: {db_path}")
        print(f"[RAG] Exists: {os.path.exists(db_path)}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./models"
        )

        self.vector_store = Chroma(
            collection_name="company_docs",
            embedding_function=self.embeddings,
            persist_directory=db_path
        )

        count = self.vector_store._collection.count()
        print(f"[RAG] Chunks in DB: {count}")

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )

    def _get_context(self, question: str) -> str:
        try:
            docs = self.retriever.invoke(question)
            if docs:
                ctx = "\n\n---\n\n".join(d.page_content for d in docs)
                print(f"[RAG] Retrieved {len(docs)} chunks, context length: {len(ctx)}")
                return ctx
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
        return ""

    def answer_query(self, question: str) -> str:
        context = self._get_context(question)

        if not context:
            print("[RAG] No context found — using general knowledge")
            template = """You are Syn, the AI assistant for Syandrix Infotech — \
a technology company.
Answer this question from your general knowledge.
Be helpful, structured, use bullet points where appropriate.
Never mention Gemini, Google, or LangChain.

Question: {question}
Answer:"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = ({"question": lambda x: x} | prompt | self.llm | StrOutputParser())
            return chain.invoke(question)

        # Context found — always use it
        print("[RAG] Context found — answering from documents")
        template = """You are Syn, the official AI assistant for Syandrix Infotech Pvt. Ltd.

You have access to Syandrix's official company documents below.
Your job is to extract and present ALL relevant information from these documents to answer the question.

STRICT RULES:
1. ALWAYS answer from the document context if ANY relevant information exists.
2. Do NOT say "I don't have information" if the answer is in the context — look carefully.
3. For company questions (CEO, services, team, contact, pricing etc.), use the document data.
4. For general tech questions, use your own knowledge AND the context if relevant.
5. Format answers with bullet points, sections, or tables where it helps clarity.
6. Never mention Gemini, Google, LangChain, or any AI technology.
7. You are Syn — Syandrix's own AI assistant.

=== SYANDRIX OFFICIAL DOCUMENTS ===
{context}
====================================

Question: {question}

Answer (use the documents above — be thorough and complete):"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": lambda _: context, "question": lambda x: x}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)


if __name__ == "__main__":
    app = RAGApplication()
    tests = [
        "who is the CEO of Syandrix?",
        "what services does Syandrix offer?",
        "tell me about Syandrix",
        "what is machine learning?",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        print(f"A: {app.answer_query(q)}")
        print("-" * 60)