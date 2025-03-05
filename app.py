import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load the YOLO model
model_path = "best_plant.pt"
try:
    model = YOLO(model_path)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit app title
st.title("Plant Disease Detection App")

# Sidebar options for input type
input_type = st.sidebar.selectbox("Select Input Type", ["Image", "Live Video", "Recorded Video"])

# Function to run YOLO detection and annotate the frame
def detect_and_annotate(frame):
    results = model(frame)
    annotated_frame = frame.copy()
    detected_diseases = []

    # Check if any objects were detected
    if results[0].boxes:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf.tolist()[0]
                cls = int(box.cls.tolist()[0])
                label = model.names[cls]

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'{label} {confidence:.2f}', 
                            (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                detected_diseases.append(f"{label} ({confidence:.2f})")
    else:
        # If no object is detected, display "Not Found"
        cv2.putText(annotated_frame, "Not Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return annotated_frame, detected_diseases

# Image Input
if input_type == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        # Read and display uploaded image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        st.image(frame, caption="Uploaded Image", use_column_width=True)

        # Run detection and display results
        annotated_frame, detected_diseases = detect_and_annotate(frame)
        st.image(annotated_frame, caption="Annotated Image", use_column_width=True)
        
        # Display detected diseases
        st.subheader("Detected Diseases")
        for disease in detected_diseases:
            st.write(disease)
    else:
        st.write("Please upload an image.")

# Live Video Input
elif input_type == "Live Video":
    st.write("Click 'Start' to begin video detection. Press 'Stop' to end.")
    
    start_video = st.button("Start")
    stop_video = st.button("Stop")
    if start_video:
        cap = cv2.VideoCapture(0)  # Start webcam capture
        stframe = st.empty()  # Placeholder for video frames
        while not stop_video:
            ret, frame = cap.read()
            if not ret:
                st.write("Error: Could not capture video.")
                break
            annotated_frame, detected_diseases = detect_and_annotate(frame)

            # Show real-time annotated frame with disease detection
            stframe.image(annotated_frame, channels="BGR")
        
        cap.release()

# Recorded Video Input
elif input_type == "Recorded Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # Save video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, detected_diseases = detect_and_annotate(frame)
            stframe.image(annotated_frame, channels="BGR")
        
        cap.release()
        st.subheader("Detected Diseases")
        for disease in detected_diseases:
            st.write(disease)
    else:
        st.write("Please upload a video.")

# Release resources when done
cv2.destroyAllWindows()
