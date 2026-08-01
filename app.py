"""
app.py -- DigitSense Streamlit Demo App.

A web-based demo for the DigitSense handwritten digit recognition model.
Upload a PNG/JPG image of a handwritten digit to see the predicted digit,
confidence scores, and a results dashboard.

Run with:
    streamlit run app.py
"""

import os
import sys
import numpy as np

# Suppress TF info/warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
from PIL import Image

# Add src/ to path for shared preprocessing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
from utils import preprocess_image

MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "digit_cnn.h5")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# -- Page config ------------------------------------------------------
st.set_page_config(
    page_title="DigitSense -- Handwritten Digit Recognition",
    page_icon="[NUM]",
    layout="wide",
)

# -- Custom CSS -------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-digit {
        font-size: 5rem;
        font-weight: bold;
        line-height: 1.2;
    }
    .confidence-text {
        font-size: 1.3rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained CNN model (cached)."""
    return tf.keras.models.load_model(MODEL_PATH)


# -- Header -----------------------------------------------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("[NUM] DigitSense")
st.markdown("**Handwritten Digit Recognition using CNN (MNIST)**")
st.markdown('</div>', unsafe_allow_html=True)

# -- Sidebar ----------------------------------------------------------
with st.sidebar:
    st.header("[UPLOAD] Upload Image")
    st.markdown("Upload a **PNG** or **JPG** image of a handwritten digit (09).")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of a single handwritten digit.",
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "DigitSense uses a Convolutional Neural Network (CNN) "
        "trained on the MNIST dataset (60,000 training images) "
        "to recognize handwritten digits 09."
    )

# -- Main content -----------------------------------------------------
if uploaded_file is not None:
    # Load and display the uploaded image
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(" Uploaded Image")
        st.image(pil_image, width=200)

    with col2:
        # Preprocess and predict
        processed, msg = preprocess_image(pil_image)

        if processed is None:
            st.error(f" {msg}")
        else:
            model = load_model()
            predictions = model.predict(processed, verbose=0)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class])

            # Prediction display
            st.markdown(
                f'<div class="prediction-box">'
                f'<div class="prediction-digit">{predicted_class}</div>'
                f'<div class="confidence-text">'
                f'Confidence: {confidence*100:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Confidence bar chart for all digits
            st.subheader("[CHART] Confidence Scores")
            chart_data = {str(i): float(predictions[0][i]) for i in range(10)}
            st.bar_chart(chart_data)

    st.markdown("---")

# -- Results Dashboard ------------------------------------------------
st.header("[CHART] Model Performance Dashboard")

tab1, tab2, tab3 = st.tabs(["Training Curves", "Confusion Matrix", "Sample Predictions"])

with tab1:
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    if os.path.exists(curves_path):
        st.image(curves_path, caption="Training & Validation Accuracy/Loss per Epoch")
    else:
        st.info("Training curves not available. Run `python src/train.py` first.")

with tab2:
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix on MNIST Test Set (10,000 images)")
    else:
        st.info("Confusion matrix not available. Run `python src/train.py` first.")

with tab3:
    preds_path = os.path.join(RESULTS_DIR, "sample_predictions.png")
    if os.path.exists(preds_path):
        st.image(preds_path, caption="Sample Predictions (green = correct, red = incorrect)")
    else:
        st.info("Sample predictions not available. Run `python src/train.py` first.")

# -- Metrics display --------------------------------------------------
metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
if os.path.exists(metrics_path):
    with st.expander("[LIST] Detailed Metrics", expanded=False):
        with open(metrics_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
