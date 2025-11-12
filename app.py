import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Make sure best.pt is in the root directory

model = load_model()

# Compliance mapping
compliance_map = {
    'Hardhat': 'Hardhat Worn',
    'Safety Vest': 'Safety Vest Worn',
    'Mask': 'Mask Worn',
    'NO-Hardhat': 'Missing Hardhat',
    'NO-Safety Vest': 'Missing Safety Vest',
    'NO-Mask': 'Missing Mask',
    'Person': 'Worker Detected',
    'machinery': 'Machinery',
    'vehicle': 'Vehicle',
    'Safety Cone': 'Safety Cone'
}

# Title
st.title("🦺 Construction PPE Compliance Detector")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference
    results = model.predict(np.array(image), verbose=False)
    boxes = results[0].boxes
    names = model.names

    # Annotate image
    annotated_img = np.array(image).copy()
    summary = {}

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        compliance_status = compliance_map.get(label, label)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0) if 'Missing' not in compliance_status else (255, 0, 0)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, compliance_status, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        summary[compliance_status] = summary.get(compliance_status, 0) + 1

    st.image(annotated_img, caption="Detection Results", use_column_width=True)

    # Show detection summary
    st.subheader("Detection Summary")
    for label, count in summary.items():
        st.write(f"- {label}: {count}")
