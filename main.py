import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import DetrImageProcessor, DetrForObjectDetection

# Load Hugging Face model for object detection
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Define the Streamlit app
def main():
    st.title("Eye Movement Detection")
    st.write("Upload an image, and the system will analyze eye direction.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Analyzing...")

        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()
        ])
        input_image = transform(image).unsqueeze(0)

        # Perform inference
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-process results
        results = processor.post_process_object_detection(outputs, target_sizes=[image.size])[0]
        labels = results["labels"]
        boxes = results["boxes"]
        scores = results["scores"]

        # Detect eyes and determine direction
        eye_directions = analyze_eyes(image, boxes, labels)
        st.write(f"Eye direction: {eye_directions}")

def analyze_eyes(image, boxes, labels):
    # This function will analyze the bounding boxes to determine the eye direction
    # For simplicity, this example uses placeholder logic
    # Replace this logic with a model trained specifically for eye direction detection
    directions = []
    for label, box in zip(labels, boxes):
        if label == "eye":  # Replace with actual eye label
            x_min, y_min, x_max, y_max = box
            eye_center_x = (x_min + x_max) / 2
            eye_center_y = (y_min + y_max) / 2

            if eye_center_x < 0.4:
                directions.append("Left")
            elif eye_center_x > 0.6:
                directions.append("Right")
            elif eye_center_y < 0.4:
                directions.append("Up")
            elif eye_center_y > 0.6:
                directions.append("Down")
            else:
                directions.append("Center")
    return directions

if __name__ == "__main__":
    main()
