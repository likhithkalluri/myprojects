import cv2
import numpy as np
import dlib
from math import hypot

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Font for displaying text
font = cv2.FONT_HERSHEY_PLAIN

# Function to calculate the midpoint between two points
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Function to calculate the blinking ratio
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

# Function to calculate gaze ratio
def get_gaze_ratio(eye_points, facial_landmarks, gray_frame):
    eye_region = np.array(
        [(facial_landmarks.part(eye_points[i]).x, facial_landmarks.part(eye_points[i]).y) for i in range(6)],
        np.int32,
    )
    mask = np.zeros_like(gray_frame)
    cv2.fillPoly(mask, [eye_region], 255)

    eye = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]

    # Inverted binary threshold for better pupil detection
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)

    height, width = threshold_eye.shape
    left_side = threshold_eye[:, :width // 2]
    right_side = threshold_eye[:, width // 2:]

    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)

    if left_white + right_white == 0:
        return 1
    return left_white / (left_white + right_white)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Blinking detection
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0), 3)

        # Gaze detection
        left_gaze_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray)
        right_gaze_ratio = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray)
        gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2

        if gaze_ratio < 0.4:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 255, 0), 3)
        elif 0.4 < gaze_ratio < 0.6:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 255, 0), 3)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
