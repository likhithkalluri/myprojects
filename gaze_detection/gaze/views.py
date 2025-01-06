from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
import cv2
import numpy as np
import dlib

# Initialize Dlib and webcam
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

# Helper function for gaze detection
def get_gaze_direction(eye_points, facial_landmarks, gray):
    eye_region = np.array([(facial_landmarks.part(eye_points[i]).x, facial_landmarks.part(eye_points[i]).y) 
                           for i in range(6)], np.int32)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    h, w = threshold_eye.shape
    left_side = threshold_eye[:, :w // 2]
    right_side = threshold_eye[:, w // 2:]
    top_side = threshold_eye[:h // 2, :]
    bottom_side = threshold_eye[h // 2:, :]

    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)
    top_white = cv2.countNonZero(top_side)
    bottom_white = cv2.countNonZero(bottom_side)

    horizontal_ratio = left_white / (left_white + right_white) if (left_white + right_white) > 0 else 0.5
    vertical_ratio = top_white / (top_white + bottom_white) if (top_white + bottom_white) > 0 else 0.5

    return horizontal_ratio, vertical_ratio

# Gaze detection view
def detect_gaze(request):
    ret, frame = cap.read()
    if not ret:
        return JsonResponse({'error': 'Camera not accessible'}, status=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    gaze_direction = "CENTER"  # Default value if no face is detected

    for face in faces:
        landmarks = predictor(gray, face)
        left_horizontal, left_vertical = get_gaze_direction([36, 37, 38, 39, 40, 41], landmarks, gray)
        right_horizontal, right_vertical = get_gaze_direction([42, 43, 44, 45, 46, 47], landmarks, gray)

        horizontal_ratio = (left_horizontal + right_horizontal) / 2
        vertical_ratio = (left_vertical + right_vertical) / 2

        if horizontal_ratio < 0.4:
            gaze_direction = "RIGHT"
        elif horizontal_ratio > 0.6:
            gaze_direction = "LEFT"
        elif vertical_ratio < 0.5:
            gaze_direction = "UP"
        else:
            gaze_direction = "DOWN"

    # Return the gaze direction as JSON
    return JsonResponse({'gaze_direction': gaze_direction})

# Webcam video feed generator
def video_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# View to serve the video feed
def stream_video(request):
    return StreamingHttpResponse(video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')

# Index page
def index(request):
    return render(request, 'gaze/index.html')