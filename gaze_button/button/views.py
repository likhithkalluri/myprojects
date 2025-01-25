from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
import cv2
import numpy as np
import dlib
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = None  # Global variable to control the camera

# Helper function to detect the pupil location
def get_pupil_location(eye_points, facial_landmarks, gray):
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
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    _, threshold_eye = cv2.threshold(blurred_eye, 30, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(max_contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx + min_x, cy + min_y)  # Adjust relative to the original image
    return None

# Improved angle calculation for horizontal gaze
def calculate_horizontal_gaze_and_angle(pupil_position, eye_region):
    eye_center_x = (np.min(eye_region[:, 0]) + np.max(eye_region[:, 0])) // 2
    eye_width = np.max(eye_region[:, 0]) - np.min(eye_region[:, 0])

    if pupil_position[0] < eye_center_x:
        direction = "right"
        angle = math.degrees(math.atan((eye_center_x - pupil_position[0]) / eye_width))
    elif pupil_position[0] > eye_center_x:
        direction = "left"
        angle = math.degrees(math.atan((pupil_position[0] - eye_center_x) / eye_width))
    else:
        direction = "center"
        angle = 0

    return direction, round(angle, 2)

# Gaze detection for multiple persons
def detect_gaze(request):
    global cap
    ret, frame = cap.read()
    if not ret:
        return JsonResponse({'error': 'Camera not accessible'}, status=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    gazes = []
    person_counter = 1

    for face in faces:
        landmarks = predictor(gray, face)

        left_pupil = get_pupil_location([36, 37, 38, 39, 40, 41], landmarks, gray)
        right_pupil = get_pupil_location([42, 43, 44, 45, 46, 47], landmarks, gray)

        if left_pupil and right_pupil:
            left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(37).x, landmarks.part(37).y),
                                        (landmarks.part(38).x, landmarks.part(38).y),
                                        (landmarks.part(39).x, landmarks.part(39).y),
                                        (landmarks.part(40).x, landmarks.part(40).y),
                                        (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

            right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                         (landmarks.part(43).x, landmarks.part(43).y),
                                         (landmarks.part(44).x, landmarks.part(44).y),
                                         (landmarks.part(45).x, landmarks.part(45).y),
                                         (landmarks.part(46).x, landmarks.part(46).y),
                                         (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

            left_direction, left_angle = calculate_horizontal_gaze_and_angle(left_pupil, left_eye_region)
            right_direction, right_angle = calculate_horizontal_gaze_and_angle(right_pupil, right_eye_region)

            gaze_direction = "left" if left_direction == "left" or right_direction == "left" else (
                "right" if left_direction == "right" or right_direction == "right" else "center")

            angle = max(left_angle, right_angle)

            gazes.append({
                'person': f"Person{person_counter}",
                'gaze_direction': f"{gaze_direction} ({angle}°)",
            })

        person_counter += 1

    return JsonResponse({'gazes': gazes})

# Webcam video feed generator
def video_feed():
    global cap
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

# Start camera feed
def start_camera(request):
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return JsonResponse({'error': 'Camera not accessible'}, status=500)
    return JsonResponse({'message': 'Camera started successfully'})

# Stop camera feed
def stop_camera(request):
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
    return JsonResponse({'message': 'Camera stopped successfully'})

# Stream video
def stream_video(request):
    return StreamingHttpResponse(video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')

# Index page
def index(request):
    return render(request, 'button/index.html')
