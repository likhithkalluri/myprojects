from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('detect-gaze/', views.detect_gaze, name='detect_gaze'),
    path('stream-video/', views.stream_video, name='stream_video'),
]
