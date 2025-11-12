import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw

@st.cache_resource
def load_model():
    return torch.jit.load("best_scripted.pt")

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

st.title("🦺 PPE Compliance Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        results = model(img_tensor)[0]  # adapt based on your model's output format

    draw = ImageDraw.Draw(image)
    summary = {}

    for det in results:  # adapt this loop to match your model's output
        x1, y1, x2, y2, cls_id = map(int, det[:5])  # example format
        label = model.names[int(cls_id)]
        status = compliance_map.get(label, label)
        color = "green" if "Missing" not in status else "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 10), status, fill=color)
        summary[status] = summary.get(status, 0) + 1

    st.image(image, caption="Detection Results", use_column_width=True)

    st.subheader("Detection Summary")
    for label, count in summary.items():
        st.write(f"- {label}: {count}")
