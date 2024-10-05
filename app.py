import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load your trained model
model = YOLO("best.pt")

# Function to run detection
def detect_drowsiness(frame):
    results = model(frame)
    # Process the results to identify drowsiness
    return results

# Streamlit UI
st.title("Driver Drowsiness Detection System")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_file is not None:
    # Read the uploaded video
    video = cv2.VideoCapture(uploaded_file.name)
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Call your detection function
        detection_results = detect_drowsiness(frame)

        # Display the frame with results (this part might vary based on how you want to show results)
        st.image(frame, channels="BGR", caption="Processed Frame", use_column_width=True)

    video.release()

# Run this app using the command: streamlit run app.py
