# chatbot_app/urls.py
from django.urls import path
from .views import HomeView, ChatAPIView

app_name = 'chatbot_app' # Namespace for the app

urlpatterns = [
    path('', HomeView.as_view(), name='home'),
    path('chat_api/', ChatAPIView.as_view(), name='chat_api'), # Note the trailing slash
]