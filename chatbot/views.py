
import json
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .brain import get_response, conversation_history
import tempfile

def index(request):
    return render(request, "chatbot/index.html")


@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON.", "status": "error"}, status=400)

    user_message = data.get("message", "").strip()
    if not user_message:
        return JsonResponse({"error": "Message cannot be empty.", "status": "error"}, status=400)
    if len(user_message) > 2000:
        return JsonResponse({"error": "Message too long (max 2000 chars).", "status": "error"}, status=400)

    reply = get_response(user_message)
    return JsonResponse({"reply": reply, "status": "ok"})

# ── Voice API — receives audio blob, returns transcript + answer ──────────
@csrf_exempt
@require_http_methods(["POST"])
def voice_chat(request):
    """
    Receives: audio/webm blob from browser MediaRecorder
    Returns:  { transcript, reply, status }
    """
    if not request.FILES.get("audio"):
        return JsonResponse({"error": "No audio file received.", "status": "error"}, status=400)
 
    audio_file = request.FILES["audio"]
 
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name
 
    try:
        import speech_recognition as sr
 
        recognizer = sr.Recognizer()
 
        # Convert webm → wav using pydub (handles browser audio format)
        try:
            from pydub import AudioSegment
            audio_seg = AudioSegment.from_file(tmp_path)
            wav_path  = tmp_path.replace(".webm", ".wav")
            audio_seg.export(wav_path, format="wav")
        except Exception as e:
            return JsonResponse({
                "error": f"Audio conversion failed: {str(e)}. Install pydub and ffmpeg.",
                "status": "error"
            }, status=500)
 
        # Transcribe
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
 
        try:
            # Use Google Web STT (free, needs internet once for transcription)
            transcript = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return JsonResponse({"error": "Could not understand audio. Please speak clearly.", "status": "error"})
        except sr.RequestError:
            # Fallback: Sphinx (offline, no internet needed)
            try:
                transcript = recognizer.recognize_sphinx(audio_data)
            except Exception:
                return JsonResponse({"error": "Speech recognition unavailable. Please type your question.", "status": "error"})
 
        # Get answer from RAG
        reply = get_response(transcript)
        return JsonResponse({"transcript": transcript, "reply": reply, "status": "ok"})
 
    finally:
        # Cleanup temp files
        try:
            os.unlink(tmp_path)
            if os.path.exists(tmp_path.replace(".webm", ".wav")):
                os.unlink(tmp_path.replace(".webm", ".wav"))
        except Exception:
            pass


@csrf_exempt
@require_http_methods(["POST"])
def clear_chat(request):
    conversation_history.clear()
    return JsonResponse({"status": "cleared"})


@require_http_methods(["GET"])
def health(request):
    import os
    from django.conf import settings
    vs_ready = os.path.exists(os.path.join(settings.BASE_DIR, "my_company_db"))
    return JsonResponse({"status": "ok", "vector_store": vs_ready})