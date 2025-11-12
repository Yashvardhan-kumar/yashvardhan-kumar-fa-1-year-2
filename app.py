import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

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

# Streamlit UI
st.title("🦺 PPE Compliance Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv5 inference
    results = model.predict(np.array(image), verbose=False)
    boxes = results[0].boxes
    names = model.names

    # Annotate image using Pillow
    draw = ImageDraw.Draw(image)
    summary = {}

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        status = compliance_map.get(label, label)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = "green" if "Missing" not in status else "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 10), status, fill=color)
        summary[status] = summary.get(status, 0) + 1

    st.image(image, caption="Detection Results", use_column_width=True)

    # Show detection summary
    st.subheader("Detection Summary")
    for label, count in summary.items():
        st.write(f"- {label}: {count}")
