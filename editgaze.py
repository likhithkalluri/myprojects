import cv2
import numpy as np
import dlib
from math import hypot

# Initialize webcam and Dlib's face detector and landmark predictor
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Function to calculate blinking ratio
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # Horizontal and vertical line lengths
    hor_line_length = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])
    ver_line_length = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])

    ratio = hor_line_length / ver_line_length
    return ratio

# Function to calculate gaze direction
def get_gaze_direction(eye_points, facial_landmarks, frame, gray):
    # Get eye region and mask
    eye_region = np.array([(facial_landmarks.part(eye_points[i]).x, facial_landmarks.part(eye_points[i]).y) 
                           for i in range(6)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    # Bounding box for the eye
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    # Horizontal and vertical gaze regions
    h, w = threshold_eye.shape
    left_side = threshold_eye[:, :w // 2]
    right_side = threshold_eye[:, w // 2:]
    top_side = threshold_eye[:h // 2, :]
    bottom_side = threshold_eye[h // 2:, :]

    # Count white pixels
    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)
    top_white = cv2.countNonZero(top_side)
    bottom_white = cv2.countNonZero(bottom_side)

    # Horizontal gaze ratio
    if left_white + right_white == 0:
        horizontal_ratio = 0
    else:
        horizontal_ratio = left_white / (left_white + right_white)

    # Vertical gaze ratio
    if top_white + bottom_white == 0:
        vertical_ratio = 0
    else:
        vertical_ratio = top_white / (top_white + bottom_white)

    return horizontal_ratio, vertical_ratio

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Eye blinking detection
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Check if eyes are blinking
        if blinking_ratio > 5.0:  # Adjust this threshold if necessary
            cv2.putText(frame, "BLINKING", (50, 150), font, 2, (0, 0, 255), 3)

        # Gaze direction detection
        left_horizontal, left_vertical = get_gaze_direction([36, 37, 38, 39, 40, 41], landmarks, frame, gray)
        right_horizontal, right_vertical = get_gaze_direction([42, 43, 44, 45, 46, 47], landmarks, frame, gray)

        # Average horizontal and vertical ratios
        horizontal_ratio = (left_horizontal + right_horizontal) / 2
        vertical_ratio = (left_vertical + right_vertical) / 2

        # Determine gaze direction
        gaze_direction = "CENTER"
        if horizontal_ratio < 0.4:
            gaze_direction = "RIGHT"
            if vertical_ratio < 0.4:
                gaze_direction = "RIGHT (UP)"
        elif horizontal_ratio > 0.6:
            gaze_direction = "LEFT"
            if vertical_ratio < 0.4:
                gaze_direction = "LEFT (UP)"
        elif vertical_ratio < 0.4:
            gaze_direction = "UP"

        cv2.putText(frame, gaze_direction, (50, 100), font, 2, (0, 255, 0), 3)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
