import streamlit as st
import os
import cv2
import dlib
import numpy as np

upload_folder = "uploaded_image"

# Check if the folder is available or not
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

st.header("Upload Image and Get Movement Description")
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_name = uploaded_file.name
    saved_path = os.path.join(upload_folder, file_name)

    # Save the file to the local folder
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Image successfully uploaded to {saved_path}")

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Load the uploaded image using OpenCV
    img = cv2.imread(saved_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load pre-trained face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this model

    # Detect face and landmarks
    faces = detector(gray)
    if faces:
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)

            # Calculate eye direction (basic example)
            eye_direction = right_eye[0] - left_eye[0]
            if eye_direction > 10:
                st.write("Person's eyes are looking to the right.")
            elif eye_direction < -10:
                st.write("Person's eyes are looking to the left.")
            else:
                st.write("Person's eyes are looking straight.")

            # Draw landmarks
            for n in range(36, 48):  # Eye landmarks
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

        # Display the image with landmarks
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
    else:
        st.error("No face detected in the uploaded image.")