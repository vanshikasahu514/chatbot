"""
debug_rag.py — Run this to diagnose why company info is not being found.

    python chatbot/debug_rag.py

This will show you:
1. How many chunks are in your vector store
2. What the chunks actually contain
3. What is retrieved for a test query
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_chatbot.settings")

import django
django.setup()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "my_company_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./models"
)

vs = Chroma(
    collection_name="company_docs",
    embedding_function=embeddings,
    persist_directory=DB_PATH
)

# ── Step 1: Count chunks ───────────────────────────────────────────────────
count = vs._collection.count()
print(f"\n{'='*60}")
print(f"STEP 1 — Total chunks in vector store: {count}")
print(f"{'='*60}")

if count == 0:
    print("\n❌ Vector store is EMPTY!")
    print("Fix: Run   python chatbot/ingest.py   first.")
    sys.exit(1)

# ── Step 2: Show first 3 chunks ────────────────────────────────────────────
print(f"\nSTEP 2 — First 3 chunks stored in DB:")
print(f"{'='*60}")
results = vs._collection.get(limit=3)
for i, doc in enumerate(results["documents"]):
    print(f"\n--- Chunk {i+1} ---")
    print(doc[:300])

# ── Step 3: Test retrieval with your query ─────────────────────────────────
test_queries = [
    "ceo of syandrix",
    "who is the CEO",
    "founder of syandrix",
    "company leadership",
    "services offered",
    "about syandrix",
]

print(f"\n\nSTEP 3 — Retrieval test for each query:")
print(f"{'='*60}")
retriever = vs.as_retriever(search_kwargs={"k": 3})

for query in test_queries:
    docs = retriever.invoke(query)
    print(f"\nQuery: '{query}'")
    print(f"  Found {len(docs)} chunks")
    for j, doc in enumerate(docs):
        print(f"  Chunk {j+1} preview: {doc.page_content[:150].strip()}")

print(f"\n{'='*60}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*60}")
print("\nIf chunks are found but answer is wrong → PDF content issue")
print("If 0 chunks found → re-run ingest.py")