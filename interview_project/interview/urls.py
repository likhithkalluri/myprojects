from django.urls import path
from .views import interview_view
from . import views

urlpatterns = [
    path("", interview_view, name="interview_home"),
    path('stream_video/', views.stream_video, name='stream_video'),
]
