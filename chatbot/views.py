from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .brain import get_response, conversation_history
import json


def index(request):
    return render(request, "chatbot/index.html")


@csrf_exempt
def chat(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
        if not user_message:
            return JsonResponse({"reply": "Please say something!"})
        reply = get_response(user_message)
        return JsonResponse({"reply": reply})
    return JsonResponse({"error": "POST required"}, status=405)


@csrf_exempt
def clear_chat(request):
    if request.method == "POST":
        conversation_history.clear()
        return JsonResponse({"status": "cleared"})
    return JsonResponse({"error": "POST required"}, status=405)