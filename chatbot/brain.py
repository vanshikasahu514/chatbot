import random
import re
from datetime import datetime
from django.conf import settings
from groq import Groq
# from gemini import Gemini
import google.generativeai as genai

client = Groq(api_key=settings.GROQ_API_KEY)
genai.configure(api_key=settings.GEMINI_API_KEY)
gemini_client = genai.GenerativeModel("gemini-pro")

# ── Must be at top level so views.py can import it ────────────────────────
conversation_history = []

# ── Company chatbot prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Syn, the official AI assistant for Syandrix — a forward-thinking technology company.

Your responsibilities:
- Answer ALL questions thoroughly and helpfully — tech, general knowledge, company info, coding, etc.
- For technical questions (coding, software, hardware, networking, AI, etc.), give detailed, accurate, expert-level answers.
- For company-related questions, represent Syandrix professionally and positively.
- Always stay friendly, professional, and concise unless a detailed answer is needed.
- If someone asks what you are, say: "I'm Syn, Syandrix's AI assistant. How can I help you today?"
- Never say you don't know — do your best to answer, and if truly uncertain, say "I'll look into that for you."
- Do NOT mention OpenAI, Groq, Meta, or any underlying AI technology. You are simply Syn by Syandrix.

Company info (use this when asked about Syandrix):
- Name: Syandrix
- Type: Technology Company
- Specialty: Cutting-edge software solutions, AI integration, and web development
- Motto: "Innovating Tomorrow, Today."
- Contact: contact@syandrix.com
- Website: www.syandrix.com
- Services: Custom Software Development, AI & ML Integration, Web & Mobile App Development, Cloud Solutions, IT Consulting

Tone: Professional, confident, helpful, and approachable.
"""

# ── Whole-word matcher (prevents "hi" matching inside "which", "this" etc.) ──
def match_keywords(text, keywords):
    for kw in keywords:
        # Use word boundary matching for single words, phrase matching for multi-word
        if " " in kw:
            if kw in text:
                return True
        else:
            if re.search(rf'\b{re.escape(kw)}\b', text):
                return True
    return False


# ── Local instant replies (no API call needed for these) ──────────────────
RESPONSES = {
    ("hello", "hi", "hey", "howdy"): [
        "Hello! Welcome to Syandrix. I'm Syn, your AI assistant. How can I help you today?",
        "Hi there! I'm Syn from Syandrix. What can I do for you?",
        "Hey! Great to have you here. How can Syandrix assist you today? 😊",
    ],
    ("bye", "goodbye", "see you", "later"): [
        "Goodbye! Thank you for visiting Syandrix. Have a great day! 👋",
        "See you later! Don't hesitate to reach out anytime.",
        "Bye! Syandrix is always here when you need us.",
    ],
    ("how are you", "how do you do", "how's it going"): [
        "I'm running perfectly — just like Syandrix's systems! How can I help you?",
        "All systems go! What can I assist you with today?",
    ],
    ("who are you", "what are you"): [
        "I'm Syn, the official AI assistant for Syandrix. I'm here to help with any questions you have!",
        "My name is Syn — Syandrix's intelligent assistant. Nice to meet you!",
    ],
    ("your name",): [
        "I'm Syn, the official AI assistant for Syandrix. I'm here to help with any questions you have!",
    ],
    ("what is syandrix", "about syandrix", "about your company"): [
        "Syandrix is a cutting-edge technology company specializing in software solutions, AI integration, and web development. Our motto is 'Innovating Tomorrow, Today.' Visit us at www.syandrix.com!",
    ],
    ("contact", "email", "reach you", "get in touch"): [
        "You can reach Syandrix at contact@syandrix.com or visit www.syandrix.com. We'd love to hear from you!",
    ],
    ("help", "what can you do", "commands"): [
        "I'm Syn, and I can help you with:\n• Tech questions (coding, AI, software, hardware)\n• Company info about Syandrix\n• General knowledge & advice\n• And much more — just ask!",
    ],
    ("joke", "tell me a joke", "funny"): [
        "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
        "Why did the developer go broke? Because he used up all his cache!",
        "A SQL query walks into a bar, walks up to two tables and asks… 'Can I join you?' 😄",
    ],
    ("what time", "current time"): [],  # handled specially below
    ("thank you", "thanks", "thx"): [
        "You're welcome! Syandrix is always happy to help 😊",
        "Anytime! That's what Syn is here for.",
        "Happy to assist! Is there anything else you need?",
    ],
    ("services", "what do you offer", "what does syandrix do"): [
        "Syandrix offers:\n• Custom Software Development\n• AI & Machine Learning Integration\n• Web & Mobile App Development\n• Cloud Solutions\n• IT Consulting\nWant to know more? Contact us at contact@syandrix.com!",
    ],
}


def _ask_groq(user_input: str) -> str:
    """Send message to Groq LLM and return the reply."""
    conversation_history.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversation_history,
            ],
            max_tokens=1024,
            temperature=0.6,
        )
        reply = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": reply})

    except Exception as e:
        reply = f"Sorry, I ran into an error: {str(e)}"
        conversation_history.pop()

    return reply


def get_response(user_input: str) -> str:
    text = user_input.lower().strip()

    # ── 1. Special case: current time ─────────────────────────────────────
    if re.search(r'\btime\b', text) and any(w in text for w in ("what", "current", "tell")):
        now = datetime.now().strftime("%I:%M %p")
        return f"The current time is {now} ⏰"

    # ── 2. Check local keyword rules (whole-word matching) ─────────────────
    for keywords, replies in RESPONSES.items():
        if match_keywords(text, keywords):
            if replies:
                return random.choice(replies)

    # ── 3. Everything else → Groq LLM ─────────────────────────────────────
    return _ask_groq(user_input)