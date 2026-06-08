"""
views.py  —  No changes needed from before, just make sure
             brain.py is your new one.
"""
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .brain import get_response, conversation_history


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


@csrf_exempt
@require_http_methods(["POST"])
def clear_chat(request):
    conversation_history.clear()
    return JsonResponse({"status": "cleared"})


@require_http_methods(["GET"])
def health(request):
    import os
    vs_ready = os.path.exists("my_company_db")
    return JsonResponse({"status": "ok", "vector_store": vs_ready})