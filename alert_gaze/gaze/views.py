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

    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)

    horizontal_ratio = left_white / (left_white + right_white) if (left_white + right_white) > 0 else 0.5

    return horizontal_ratio

# Calculate horizontal direction and angle
def calculate_horizontal_gaze(horizontal_ratio):
    if 0.45 <= horizontal_ratio <= 0.55:
        return 0, "center"  # Eyes are centered
    elif horizontal_ratio > 0.55:
        angle = (horizontal_ratio - 0.5) * 90
        return angle, "left"

    else:
        angle = (0.5 - horizontal_ratio) * 90
        return angle, "right"

# Gaze detection for multiple persons
def detect_gaze(request):
    ret, frame = cap.read()
    if not ret:
        return JsonResponse({'error': 'Camera not accessible'}, status=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    gazes = []
    person_counter = 1

    for face in faces:
        landmarks = predictor(gray, face)

        # Detect horizontal gaze direction
        left_horizontal = get_gaze_direction([36, 37, 38, 39, 40, 41], landmarks, gray)
        right_horizontal = get_gaze_direction([42, 43, 44, 45, 46, 47], landmarks, gray)

        horizontal_ratio = (left_horizontal + right_horizontal) / 2
        angle, direction = calculate_horizontal_gaze(horizontal_ratio)

        # Detect vertical gaze direction (up/down)
        nose_y = landmarks.part(30).y
        face_top = face.top()
        face_bottom = face.bottom()

        if nose_y < face_top + (face_bottom - face_top) * 0.4:
            vertical_gaze_direction = "up"
        elif nose_y > face_bottom - (face_bottom - face_top) * 0.4:
            vertical_gaze_direction = "down"
        else:
            vertical_gaze_direction = "straight"

        # Determine the final gaze output
        if vertical_gaze_direction != "straight":
            gaze_direction = f"Person looking {vertical_gaze_direction}"
            angle = 0  # No angle for up/down
        elif direction == "center":
            gaze_direction = "Person looking center"
        else:
            gaze_direction = f"Person looking {direction} ({angle:.1f}°)"

        gazes.append({
            'person': f"Person{person_counter}",
            'gaze_direction': gaze_direction,
            'angle': angle,
            'horizontal_direction': direction,
            'vertical_direction': vertical_gaze_direction,
            'face_position': {
                'left': face.left(),
                'top': face.top(),
                'right': face.right(),
                'bottom': face.bottom()
            }
        })

        person_counter += 1

    return JsonResponse({'gazes': gazes})

# Webcam video feed generator
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

# Stream the video feed
def stream_video(request):
    return StreamingHttpResponse(video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')

# Index page
def index(request):
    return render(request, 'gaze/index.html')