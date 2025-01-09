from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('stream_video/', views.stream_video, name='stream_video'),
    path('detect_gaze/', views.detect_gaze, name='detect_gaze'),
]
