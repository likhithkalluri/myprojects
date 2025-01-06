from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import time

# Load pre-trained face detection model (Haar Cascade classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the default webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    raise Exception("Webcam not found")

def get_face_count_and_frame():
    ret, frame = cap.read()
    if not ret:
        return None, 0
    
    #Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale (Haar Cascade requires grayscale image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Valid face count
    face_count = len(faces)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Encode frame as JPEG
    _, jpeg_frame = cv2.imencode('.jpg', frame)
    return jpeg_frame.tobytes(), face_count

def video_feed(request):
    def generate():
        while True:
            frame, face_count = get_face_count_and_frame()
            if frame:
                # This generator will continuously stream the webcam frames
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    # Get the face count
    face_count = get_face_count_and_frame()[1]

    # Render the template with the face count
    return render(request, 'face/index.html', {'face_count': face_count})

from django.http import JsonResponse

def get_face_count(request):
    face_count = get_face_count_and_frame()[1]
    return JsonResponse({'face_count': face_count})
