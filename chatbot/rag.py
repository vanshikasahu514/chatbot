"""
rag.py — Complete rewrite, path issue fixed
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import speech_recognition as sr
import pyttsx3
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# ── Absolute path so it works from any working directory ──────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "my_company_db")   # always project root

class VoiceAssistant:
    """Handles Speech-to-Text and Text-to-Speech."""

    def __init__(self):
        # --- Speech Recognition (STT) ---
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300   # mic sensitivity
        self.recognizer.pause_threshold = 1.0    # seconds of silence = end of speech

        # --- Text to Speech (TTS) ---
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 165)    # speaking speed
        self.tts_engine.setProperty("volume", 1.0)  # 0.0 to 1.0

    def listen(self) -> str | None:
        """Record from microphone and return transcribed text."""
        with sr.Microphone() as source:
            print("\n  Adjusting for ambient noise... please wait.")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("  Listening... speak your question now.")
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
            except sr.WaitTimeoutError:
                print("  No speech detected. Try again.")
                return None

        print("  Transcribing...")
        try:
            text = self.recognizer.recognize_google(audio)  # uses Google Web STT (free)
            print(f"  You said: {text}")
            return text
        except sr.UnknownValueError:
            print("  Could not understand audio. Please speak clearly.")
            return None
        except sr.RequestError as e:
            print(f"  STT service error: {e}")
            return None

    def speak(self, text: str):
        """Convert text answer to speech."""
        # pyttsx3 can stumble on special chars; clean lightly
        clean = text.replace("•", "").replace("**", "")
        self.tts_engine.say(clean)
        self.tts_engine.runAndWait()

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


def run_voice_mode(rag: RAGApplication, voice: VoiceAssistant):
    """Continuous voice Q&A loop. Say 'exit' or 'quit' to stop."""
    print("\n  Voice Mode started. Say 'exit' or 'quit' to stop.\n")
    while True:
        question = voice.listen()
        if question is None:
            continue
        if question.lower().strip() in {"exit", "quit", "stop"}:
            print("  Exiting voice mode.")
            voice.speak("Goodbye!")
            break

        print("\n  Thinking...\n")
        answer = rag.answer_query(question)
        print(f"  Answer:\n{answer}\n")
        voice.speak(answer)


def run_text_mode(rag: RAGApplication):
    print("\n⌨️  Text Mode. Type 'exit' to quit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        answer = rag.answer_query(question)
        print(f"\n bot  Answer:\n{answer}\n")

if __name__ == "__main__":
    rag = RAGApplication()
    voice=VoiceAssistant()

    choice = input("Enter 1 for Voice_assistant: ")
    if choice=='1':
        run_voice_mode(rag,voice)
    else:
        run_text_mode(rag)
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