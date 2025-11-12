import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='local')

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
st.title("🦺 PPE Compliance Detector (No OpenCV)")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference
    results = model(np.array(image))
    detections = results.pandas().xyxy[0]

    # Annotate image using Pillow
    draw = ImageDraw.Draw(image)
    summary = {}

    for _, row in detections.iterrows():
        label = row['name']
        compliance_status = compliance_map.get(label, label)
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        color = "green" if "Missing" not in compliance_status else "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 10), compliance_status, fill=color)
        summary[compliance_status] = summary.get(compliance_status, 0) + 1

    st.image(image, caption="Detection Results", use_column_width=True)

    # Show detection summary
    st.subheader("Detection Summary")
    for label, count in summary.items():
        st.write(f"- {label}: {count}")
