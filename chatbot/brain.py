"""
brain.py  —  Drop this into chatbot/brain.py
Integrates your RAGApplication with Django views.
"""
import os
import re
import random
from datetime import datetime

# ── conversation_history must exist at module level (views.py imports it) ──
conversation_history = []

# ── Lazy-load RAG so Django settings are ready before imports ──────────────
_rag_app = None

def _get_rag():
    global _rag_app
    if _rag_app is None:
        from .rag import RAGApplication          # your existing rag.py
        _rag_app = RAGApplication(db_path="my_company_db")
    return _rag_app


# ── Quick keyword replies — instant, no LLM cost ───────────────────────────
QUICK_REPLIES = {
    ("hello", "hi", "hey", "howdy"): [
        "Hello! I'm Syn, Syandrixin's AI assistant. How can I help you today?",
        "Hi there! Welcome to Syandrixin. What can I do for you? ",
    ],
    ("bye", "goodbye", "see you"): [
        "Goodbye! Have a great day! ",
        "See you later! Syandrixin is always here when you need us.",
    ],
    ("thanks", "thank you", "thx"): [
        "You're welcome! ",
        "Happy to help! Is there anything else you need?",
    ],
    ("who are you", "what are you", "your name"): [
        "I'm Syn — Syandrixin's official AI assistant, powered by RAG technology. "
        "I can answer company questions from our docs and any general tech questions!",
    ],
}


def _match(text: str, keywords: tuple) -> bool:
    """Whole-word match — prevents 'hi' matching inside 'which'."""
    for kw in keywords:
        pattern = rf"\b{re.escape(kw)}\b" if " " not in kw else re.escape(kw)
        if re.search(pattern, text):
            return True
    return False


def get_response(user_input: str) -> str:
    """
    Priority order:
      1. Current time  — instant, no API
      2. Quick keyword replies — instant, no API
      3. RAGApplication.answer_query() — your rag.py handles everything:
           • retrieves from Chroma (company docs)
           • sends context + question to Gemini
           • if no context found, Gemini answers from general knowledge
    """
    text = user_input.lower().strip()

    # ── 1. Time ───────────────────────────────────────────────────────────
    if re.search(r"\btime\b", text) and any(w in text for w in ("what", "current", "now", "tell")):
        return f"The current time is {datetime.now().strftime('%I:%M %p')} "

    # ── 2. Quick replies ──────────────────────────────────────────────────
    for keywords, replies in QUICK_REPLIES.items():
        if _match(text, keywords):
            return random.choice(replies)

    # ── 3. RAG chain (Chroma retrieval → Gemini) ─────────────────────────
    try:
        rag = _get_rag()
        return rag.answer_query(user_input)
    except Exception as e:
        return f"Sorry, I ran into an error: {str(e)}"