from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('get_face_count/', views.get_face_count, name='get_face_count'),  # Add this line
]
