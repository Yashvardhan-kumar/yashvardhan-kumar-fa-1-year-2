import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

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
    results = model(np.array(image))
    detections = results.pandas().xyxy[0]

    # Annotate image
    annotated_img = np.array(image).copy()
    for _, row in detections.iterrows():
        label = row['name']
        compliance_status = compliance_map.get(label, label)
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        color = (0, 255, 0) if 'Missing' not in compliance_status else (255, 0, 0)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, compliance_status, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    st.image(annotated_img, caption="Detection Results", use_column_width=True)

    # Show detection summary
    st.subheader("Detection Summary")
    for label in detections['name'].unique():
        count = (detections['name'] == label).sum()
        st.write(f"- {compliance_map.get(label, label)}: {count}")
