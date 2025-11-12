import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np

# ============================
# Page Config & DARK THEME CSS
# ============================
st.set_page_config(
    page_title="PPE Compliance Detector",
    page_icon="Hard hat",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background: #0e1117;
        color: #e2e8f0;
    }
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    .header-title {
        font-size: 3rem !important;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Upload Box - Dark */
    .upload-box {
        border: 3px dashed #3b82f6;
        border-radius: 15px;
        padding: 2.5rem;
        text-align: center;
        background-color: #1e293b;
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        border-color: #60a5fa;
        background-color: #334155;
        transform: translateY(-2px);
    }

    /* Cards */
    .summary-card {
        background: #1e293b;
        padding: 1.8rem;
        border-radius: 14px;
        border: 1px solid #334155;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        margin: 1.5rem 0;
    }

    /* Text Colors */
    .compliant { color: #34d399; font-weight: bold; }
    .non-compliant { color: #f87171; font-weight: bold; }
    .neutral { color: #a78bfa; font-weight: bold; }

    /* Metrics */
    .stMetric > div {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #334155;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        color: #64748b;
        font-size: 0.9rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================
# Load Model
# ============================
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

model = load_model()

# ============================
# Compliance Map
# ============================
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

# ============================
# Header
# ============================
st.markdown('<h1 class="header-title">Hard hat PPE Compliance Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Real-time PPE detection for construction site safety monitoring</p>', unsafe_allow_html=True)

# ============================
# Upload Area
# ============================
st.markdown("""
<div class="upload-box">
    <h3 style="color:#60a5fa;">Upload Site Image</h3>
    <p style="color:#94a3b8;">JPG • JPEG • PNG</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    
    with st.spinner("Analyzing with YOLOv5..."):
        results = model(image)
        df = results.pandas().xyxy[0]
        
        annotated_img = np.array(image)
        for _, row in df.iterrows():
            label = row['name']
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            is_compliant = label in ['Hardhat', 'Safety Vest', 'Mask']
            color = (0, 255, 0) if is_compliant else (255, 0, 0)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            text = compliance_map.get(label, label)
            cv2.putText(annotated_img, text, (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    with col2:
        st.markdown("### Detection Result")
        st.image(annotated_img, use_column_width=True)

    # ============================
    # Summary Card
    # ============================
    st.markdown("### Compliance Summary")
    st.markdown("<div class='summary-card'>", unsafe_allow_html=True)

    compliant_count = 0
    violations = 0
    workers = len(df[df['name'] == 'Person'])

    for label in df['name'].unique():
        count = (df['name'] == label).sum()
        text = compliance_map.get(label, label)
        if any(x in text for x in ['Hardhat Worn', 'Vest Worn', 'Mask Worn']):
            st.markdown(f"<p class='compliant'>Hardhat {text}: <strong>{count}</strong></p>", unsafe_allow_html=True)
            compliant_count += count
        elif 'Missing' in text:
            st.markdown(f"<p class='non-compliant'>Warning {text}: <strong>{count}</strong></p>", unsafe_allow_html=True)
            violations += count
        else:
            st.markdown(f"<p class='neutral'>{text}: <strong>{count}</strong></p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================
    # Metrics
    # ============================
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Workers", workers)
    with c2:
        st.metric("Compliant Items", compliant_count)
    with c3:
        st.metric("Violations", violations, delta=violations)

    # ============================
    # Final Status
    # ============================
    if violations == 0 and workers > 0:
        st.success("**All workers are fully compliant!** Excellent safety standards.")
    elif violations > 0:
        st.error(f"**{violations} PPE violation(s) detected!** Take immediate action.")

else:
    st.info("Upload an image to start PPE compliance checking.")

# ============================
# Footer
# ============================
st.markdown("""
<div class="footer">
    <p>Hard hat PPE Compliance Detector • YOLOv5 + Streamlit</p>
    <p>Keeping construction sites safe, 24/7</p>
</div>
""", unsafe_allow_html=True)
