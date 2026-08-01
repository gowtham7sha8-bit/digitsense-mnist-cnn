"""
predict.py — DigitSense CLI prediction script.

Usage:
    python src/predict.py <image_path>

Loads the trained CNN model and predicts the handwritten digit in the
given image. Uses the shared preprocessing pipeline from utils.py.
"""

import sys
import os
import numpy as np

# Suppress TF info/warning logs (CPU/oneDNN noise)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from PIL import Image

# Add src/ to path so we can import utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from utils import preprocess_image

PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "model", "digit_cnn.h5")


def predict(image_path):
    """
    Predict the digit in the given image file.

    Parameters
    ----------
    image_path : str
        Path to the image file (PNG or JPG).

    Returns
    -------
    int or None
        Predicted digit (0-9), or None if no digit detected.
    float
        Confidence (0.0-1.0).
    str
        Status message.
    """
    if not os.path.exists(image_path):
        return None, 0.0, f"Error: File not found: {image_path}"

    if not os.path.exists(MODEL_PATH):
        return None, 0.0, f"Error: Model not found at {MODEL_PATH}. Run train.py first."

    # Open image with PIL, then pass to shared preprocessing
    pil_image = Image.open(image_path)
    processed, msg = preprocess_image(pil_image)

    if processed is None:
        return None, 0.0, msg

    # Load model and predict
    model = tf.keras.models.load_model(MODEL_PATH)
    predictions = model.predict(processed, verbose=0)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])

    return predicted_class, confidence, ""


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        print("Example: python src/predict.py results/sample_digit_0.png")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Predicting digit in: {image_path}")

    predicted_digit, confidence, message = predict(image_path)

    if predicted_digit is None:
        print(f"  {message}")
    else:
        print(f"  Predicted digit: {predicted_digit}")
        print(f"  Confidence:      {confidence:.4f} ({confidence*100:.1f}%)")


if __name__ == "__main__":
    main()
