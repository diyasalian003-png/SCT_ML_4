"""
Test the trained gesture recognition model on a single image.

Usage:
    python predict_gesture.py path/to/your/photo.png
"""

import sys
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (64, 64)

# Must match the order label_names was built in during training
LABEL_NAMES = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down",
]


def predict(image_path):
    model = tf.keras.models.load_model("outputs/gesture_model.keras")

    img = Image.open(image_path).convert("L").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    probs = model.predict(arr, verbose=0)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx] * 100

    print(f"\nImage: {image_path}")
    print(f"Predicted gesture: {LABEL_NAMES[pred_idx]}")
    print(f"Confidence: {confidence:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_gesture.py path/to/your/photo.png")
    else:
        predict(sys.argv[1])
