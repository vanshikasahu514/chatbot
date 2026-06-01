"""
ingest.py  —  Your existing file + txt file support added.

Run once (and re-run whenever you add new docs):
    python chatbot/ingest.py

Supports: .pdf, .txt files in chatbot/Documents/
"""
import os
import sys

# Bootstrap Django so DJANGO_SETTINGS_MODULE is set before any app imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_chatbot.settings")

import django
django.setup()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


class RAGIngestor:
    def __init__(self, db_path="my_company_db"):
        self.db_path = db_path

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./models"
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def _get_vector_store(self):
        return Chroma(
            collection_name="company_docs",
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )

    def ingest_pdf(self, file_path: str):
        """Ingest a single PDF file."""
        print(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        self._get_vector_store().add_documents(chunks)
        print(f"  Ingested {len(chunks)} chunks from {os.path.basename(file_path)} ✅")

    def ingest_txt(self, file_path: str):
        """Ingest a single TXT file."""
        print(f"Loading TXT: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        self._get_vector_store().add_documents(chunks)
        print(f"  Ingested {len(chunks)} chunks from {os.path.basename(file_path)} ✅")

    def ingest_folder(self, folder_path: str):
        """Ingest ALL .pdf and .txt files in a folder."""
        if not os.path.exists(folder_path):
            print(f"[ERROR] Folder not found: {folder_path}")
            sys.exit(1)

        files = os.listdir(folder_path)
        pdfs = [f for f in files if f.endswith(".pdf")]
        txts = [f for f in files if f.endswith(".txt")]

        if not pdfs and not txts:
            print(f"[ERROR] No .pdf or .txt files found in {folder_path}")
            sys.exit(1)

        for fname in pdfs:
            self.ingest_pdf(os.path.join(folder_path, fname))
        for fname in txts:
            self.ingest_txt(os.path.join(folder_path, fname))

        print(f"\nAll documents ingested into '{self.db_path}' ✅")
        print("You can now run: python manage.py runserver")


if __name__ == "__main__":
    ingestor = RAGIngestor()

    # ── Option A: ingest your existing PDF (original behaviour) ───────────
    ingestor.ingest_pdf("chatbot/Documents/syandrix_infotech_knowledge_base.pdf")

    # ── Option B: ingest an entire folder of PDFs + TXTs ──────────────────
    # ingestor.ingest_folder("chatbot/Documents")