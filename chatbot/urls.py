from django.urls import path
from . import views

urlpatterns = [
    path("",              views.index,       name="index"),
    path("api/chat/",     views.chat,        name="api_chat"),
    path("api/voice/",    views.voice_chat,  name="api_voice"),   
    path("api/clear/",    views.clear_chat,  name="api_clear"),
    path("api/health/",   views.health,      name="api_health"),
]