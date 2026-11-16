import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# CONFIGURATION
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "resnet_96x1024.keras"
MAPPING_PATH = BASE_DIR / "model" / "class_mapping.json"

OUT_H = 96
OUT_W = 1024
WORK = 2048
THRESHOLD = 0.40

# LOAD MODEL
def load_predictor():
    """
    Load trained ResNet model and class mapping.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping tidak ditemukan: {MAPPING_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    classes = mapping["classes"]
    class_indices = mapping["c2i"]

    # Validate class mapping
    for index, label in enumerate(classes):
        if class_indices.get(label) != index:
            raise ValueError(
                f"Mapping kelas tidak sesuai pada {label}"
            )

    # Validate model output
    output_classes = model.output_shape[-1]

    if output_classes != len(classes):
        raise ValueError(
            f"Jumlah output model ({output_classes}) "
            f"tidak sama dengan jumlah kelas ({len(classes)})"
        )

    return model, classes

# IMAGE PREPROCESSING

def preprocess_image(image_bytes):
    """
    Convert uploaded image into model input format.
    """

    # Validate image input
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ValueError("Input harus berupa bytes gambar.")

    if len(image_bytes) == 0:
        raise ValueError("File gambar kosong.")

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if bgr is None:
        raise ValueError("Gambar tidak dapat dibaca.")

    # Convert to grayscale
    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    # Resize image for processing
    scale = min(
        1.0,
        WORK / max(gray.shape)
    )

    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    # Background normalization
    background = cv2.medianBlur(gray, 51)

    gray = cv2.divide(
        gray,
        background,
        scale=255
    )

    # Detect handwritten ink
    _, ink = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    ys, xs = np.where(ink > 0)

    # Crop handwritten area
    if len(xs) > 100:
        gray = gray[
            ys.min():ys.max() + 1,
            xs.min():xs.max() + 1
        ]

    height, width = gray.shape

    # Maintain aspect ratio
    scale = min(
        OUT_H / height,
        OUT_W / width
    )

    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))

    gray = cv2.resize(
        gray,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Add white padding
    padded = np.full(
        (OUT_H, OUT_W),
        255,
        dtype=np.uint8
    )

    y_offset = (OUT_H - new_height) // 2
    x_offset = (OUT_W - new_width) // 2

    padded[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = gray

    original_rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )

    return original_rgb, padded

# PREDICTION

def predict_image(model, classes, processed_image):
    """
    Predict handwriting writer using ResNet.
    """

    # Validate processed image
    if processed_image is None:
        raise ValueError("Processed image tidak boleh kosong.")

    if processed_image.shape != (OUT_H, OUT_W):
        raise ValueError(
            f"Ukuran gambar harus {(OUT_H, OUT_W)}, "
            f"tetapi menerima {processed_image.shape}"
        )

    # Convert grayscale image into 3 channels
    image_input = np.repeat(
        processed_image[None, :, :, None],
        3,
        axis=-1
    ).astype("float32")

    # Run model prediction
    probabilities = model.predict(
        image_input,
        verbose=0
    )[0]

    # Validate prediction output
    if len(probabilities) != len(classes):
        raise ValueError(
            "Jumlah hasil prediksi tidak sesuai dengan mapping kelas."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Hasil prediksi mengandung nilai tidak valid.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Nilai probabilitas harus berada di antara 0 dan 1.")

    # Get top 3 predictions
    top_indices = probabilities.argsort()[::-1][:3]

    top_predictions = []

    for index in top_indices:
        top_predictions.append({
            "label": classes[index],
            "confidence": float(probabilities[index])
        })

    best_prediction = top_predictions[0]

    result = {
        "predicted_class": best_prediction["label"],
        "confidence": best_prediction["confidence"],
        "is_confident": best_prediction["confidence"] >= THRESHOLD,
        "top_3": top_predictions
    }

    return result
