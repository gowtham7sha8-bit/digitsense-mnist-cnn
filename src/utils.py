"""
utils.py -- Shared image preprocessing for DigitSense.

Provides a single preprocessing function used by both predict.py and app.py
to ensure identical treatment of input images.
"""

import numpy as np
from PIL import Image, ImageOps


def preprocess_image(pil_image):
    """
    Preprocess a PIL Image for digit prediction.

    Accepts an already-opened PIL Image object (not a file path).
    Returns a numpy array of shape (1, 28, 28, 1) normalized to [0, 1],
    or None if no digit is detected.

    Steps:
        1. Convert to grayscale.
        2. Invert if background is light (avg pixel > 127).
        3. Threshold and crop to bounding box of non-zero pixels.
        4. Resize longest side to 20px (preserving aspect ratio),
           center on a 28x28 black canvas (mirrors MNIST construction).
        5. Normalize to [0, 1] and reshape to (1, 28, 28, 1).

    Parameters
    ----------
    pil_image : PIL.Image.Image
        An already-opened PIL image of a handwritten digit.

    Returns
    -------
    np.ndarray or None
        Preprocessed image array ready for model.predict(), or None
        if the image appears blank (no digit detected).
    str
        Status message -- empty string on success, error description
        if no digit was detected.
    """
    # 1. Convert to grayscale
    img = pil_image.convert("L")

    # 2. Invert if light background
    img_array = np.array(img)
    if img_array.mean() > 127:
        img = ImageOps.invert(img)
        img_array = np.array(img)

    # 3. Threshold and find bounding box
    # Pixels > 30 count as "digit"
    threshold = 30
    binary = (img_array > threshold).astype(np.uint8) * 255
    binary_img = Image.fromarray(binary)
    bbox = binary_img.getbbox()

    if bbox is None:
        # No non-zero pixels found -- blank or near-blank image
        return None, "No digit detected -- the image appears blank or has insufficient contrast."

    # Crop to bounding box
    cropped = img.crop(bbox)

    # 4. Resize so longest side is 20px, preserving aspect ratio
    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(20 * h / w)))
    else:
        new_h = 20
        new_w = max(1, int(round(20 * w / h)))

    cropped_resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    # Center on a 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas.paste(cropped_resized, (x_offset, y_offset))

    # 5. Normalize to [0, 1] and reshape to (1, 28, 28, 1)
    img_array = np.array(canvas).astype(np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array, ""
