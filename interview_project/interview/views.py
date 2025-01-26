from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
import cv2


# Helper functions to generate reusable components


cap = cv2.VideoCapture(0)

def generate_buttons():
    return [
        {"icon": "fa fa-gear", "text": "Settings", "is_primary": False},  # Settings
        {"icon": "fas fa-microphone-slash", "text": "Mute", "is_primary": False},  # Mute
        {"icon": "", "text": "Start Interview", "is_primary": True},  # Start Interview
        {"icon": "fas fa-video", "text": "Camera", "is_primary": False},  # Camera
        {"icon": "fas fa-code", "text": "Code Editor", "is_primary": False},  # Code Editor
    ]

def generate_tabs():
    return [
        {"text": "Conversation", "is_active": True},
        {"text": "Interview Details", "is_active": False},
    ]

# View to render the reusable UI
def interview_view(request):
    context = {
        "header_text": "Interview on Python - Sudarsan",
        "logo_url": "https://app.reaidy.io/static/media/logo.0b902054ab86a6253951.png",
        "video_thumbnail": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80",
        "buttons": generate_buttons(),
        "tabs": generate_tabs(),
    }
    return render(request, "interview/home.html", context)


def video_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for display purposes only
        display_frame = cv2.flip(frame, 1)

        _, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def stream_video(request):
    return StreamingHttpResponse(video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')