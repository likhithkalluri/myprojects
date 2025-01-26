from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('stream_video/', views.stream_video, name='stream_video'),
    path('detect_gaze/', views.detect_gaze, name='detect_gaze'),
    path('start_camera/', views.start_camera, name='start_camera'),
    path('stop_camera/', views.stop_camera, name='stop_camera'),
    path('result_page/', views.result_page, name='result_page'),  # New result page URL
]
