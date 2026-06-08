"""
ingest.py — Fixed with absolute paths so db always saves to project root
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_chatbot.settings")

import django
django.setup()

from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# ── Always use absolute path — works from any working directory ────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "my_company_db")
PDF_PATH = os.path.join(BASE_DIR, "chatbot", "Documents", "Syandrix Chatbot Training-V2.0.pdf")

print(f"Project root : {BASE_DIR}")
print(f"DB will save to: {DB_PATH}")
print(f"PDF path: {PDF_PATH}")


class RAGIngestor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./models"
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def clear_all(self):
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            print(f"✅ Deleted old DB at: {DB_PATH}")
        else:
            print(f"No existing DB found at: {DB_PATH}")

    def _get_vector_store(self):
        return Chroma(
            collection_name="company_docs",
            embedding_function=self.embeddings,
            persist_directory=DB_PATH      # absolute path
        )

    def clean_text(self, text: str) -> str:
        noise_lines = [
            "Syandrix Infotech Pvt. Ltd. | Enterprise Chatbot Training Document v2.0 | Confidential",
            "Syandrix Infotech Pvt. Ltd. — Internal Document — Not for External Distribution",
        ]
        for line in noise_lines:
            text = text.replace(line, "")
        return text.strip()

    def ingest_pdf(self, file_path: str):
        print(f"\nLoading: {file_path}")
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False

        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        print(f"  Pages loaded: {len(docs)}")

        for doc in docs:
            doc.page_content = self.clean_text(doc.page_content)

        docs = [d for d in docs if len(d.page_content.strip()) > 50]
        print(f"  Pages after cleaning: {len(docs)}")

        chunks = self.splitter.split_documents(docs)
        print(f"  Chunks created: {len(chunks)}")

        if chunks:
            print(f"\n  First chunk preview:")
            print(f"  {'-'*50}")
            print(f"  {chunks[0].page_content[:300]}")
            print(f"  {'-'*50}")

        self._get_vector_store().add_documents(chunks)

        final_count = self._get_vector_store()._collection.count()
        print(f"\n  ✅ DB saved to: {DB_PATH}")
        print(f"  ✅ Total chunks in DB: {final_count}")
        return True


if __name__ == "__main__":
    ingestor = RAGIngestor()

    print("\n" + "="*60)
    print("STEP 1 — Clearing old vector store")
    print("="*60)
    ingestor.clear_all()

    print("\n" + "="*60)
    print("STEP 2 — Ingesting PDF")
    print("="*60)
    success = ingestor.ingest_pdf(PDF_PATH)

    if success:
        print("\n" + "="*60)
        print("✅ DONE! Now run: python manage.py runserver")
        print("="*60)
    else:
        print("\n❌ Ingest failed — check file path above")