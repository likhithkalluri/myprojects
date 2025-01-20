# gaze/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Home page
    path('detect_gaze/', views.detect_gaze, name='detect_gaze'),  # Gaze detection endpoint
    path('stream_video/', views.stream_video, name='stream_video'),  # Video feed endpoint
]
