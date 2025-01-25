from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),                     # For serving the frontend
    path('stream_video/', views.stream_video, name='stream_video'),  # For video streaming
    path('detect_gaze/', views.detect_gaze, name='detect_gaze'),      # For gaze detection
    path('start_camera/', views.start_camera, name='start_camera'),  # To start the camera
    path('stop_camera/', views.stop_camera, name='stop_camera'),     # To stop the camera
]
