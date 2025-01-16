# Human-Pose-Estimation
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# Title of the app
st.title("HUMAN POSE ESTIMAION APP")
st.write("Upload an image, and  The Model will estimate the pose.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Convert the image from RGB to BGR (required for OpenCV and MediaPipe)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Process the image with MediaPipe Pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        # Draw pose landmarks on the image
        if results.pose_landmarks:
            annotated_image = image_bgr.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Landmarks
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # Connections
            )

            # Convert BGR to RGB for display in Streamlit
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # Display the pose-estimated image
            st.image(annotated_image_rgb, caption="Pose Estimation Output", use_column_width=True)
        else:
            st.write("No pose landmarks detected in the uploaded image.")
