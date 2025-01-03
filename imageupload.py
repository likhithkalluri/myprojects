import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

# Streamlit UI
st.header("Accurate Eye Movement Analysis")
uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Mediapipe setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,  # Dynamic mode for real-time processing
        max_num_faces=1, 
        refine_landmarks=True)

    # Process the image
    results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Eye landmark indices
            LEFT_EYE_INDICES = [33, 133, 160, 144, 153, 154, 155, 173, 246]
            RIGHT_EYE_INDICES = [362, 263, 387, 373, 380, 381, 382, 384, 385]

            height, width, _ = image_np.shape

            def get_eye_landmarks(indices):
                """Extracts the eye landmarks and converts them to pixel coordinates."""
                return np.array([[int(face_landmarks.landmark[i].x * width), 
                                  int(face_landmarks.landmark[i].y * height)] for i in indices])

            left_eye_points = get_eye_landmarks(LEFT_EYE_INDICES)
            right_eye_points = get_eye_landmarks(RIGHT_EYE_INDICES)

            # Calculate the iris center by using the average position of the inner and outer corners of the eye
            def get_iris_center(eye_points):
                # Iris center is approximated by the average of the eye corners
                x_center = int((eye_points[0][0] + eye_points[3][0]) / 2)
                y_center = int((eye_points[1][1] + eye_points[5][1]) / 2)
                return x_center, y_center

            left_iris_center = get_iris_center(left_eye_points)
            right_iris_center = get_iris_center(right_eye_points)

            # Calculate the gaze direction based on the iris center relative to eye bounding box
            def get_gaze_direction(iris_center, x_min, y_min, x_max, y_max):
                """Determines horizontal and vertical gaze position."""
                x_ratio = (iris_center[0] - x_min) / (x_max - x_min)
                y_ratio = (iris_center[1] - y_min) / (y_max - y_min)
                return x_ratio, y_ratio

            # Get bounding boxes for each eye
            def get_eye_bbox(points):
                """Find the bounding box of an eye."""
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                return x_min, y_min, x_max, y_max

            left_x_min, left_y_min, left_x_max, left_y_max = get_eye_bbox(left_eye_points)
            right_x_min, right_y_min, right_x_max, right_y_max = get_eye_bbox(right_eye_points)

            # Get the gaze directions
            left_gaze_x, left_gaze_y = get_gaze_direction(left_iris_center, left_x_min, left_y_min, left_x_max, left_y_max)
            right_gaze_x, right_gaze_y = get_gaze_direction(right_iris_center, right_x_min, right_y_min, right_x_max, right_y_max)

            # Combine gaze direction for both eyes
            avg_gaze_x = (left_gaze_x + right_gaze_x) / 2
            avg_gaze_y = (left_gaze_y + right_gaze_y) / 2

            # Determine gaze direction based on thresholds
            if avg_gaze_x < 0.4:
                gaze_horizontal = "Looking Left"
            elif avg_gaze_x > 0.6:
                gaze_horizontal = "Looking Right"
            else:
                gaze_horizontal = "Looking Straight"

            if avg_gaze_y < 0.4:
                gaze_vertical = "Looking Up"
            elif avg_gaze_y > 0.6:
                gaze_vertical = "Looking Down"
            else:
                gaze_vertical = "Looking Straight"

            # Final output
            st.success(f"Horizontal Gaze: {gaze_horizontal}")
            st.success(f"Vertical Gaze: {gaze_vertical}")
            
            # Optional: Draw landmarks and gaze direction on the image
            image_with_landmarks = image_np.copy()
            for point in left_eye_points:
                cv2.circle(image_with_landmarks, tuple(point), 2, (0, 255, 0), -1)
            for point in right_eye_points:
                cv2.circle(image_with_landmarks, tuple(point), 2, (0, 255, 0), -1)
            st.image(image_with_landmarks, caption="Detected Eye Landmarks", use_column_width=True)

    else:
        st.error("No face landmarks detected. Please try another image.")
