"""
predictor.py — ML Prediction & OpenCV Feature Extraction
=========================================================
Handles model loading, image prediction, CV feature extraction,
and mock prediction fallback when the model file is unavailable.

Model: ResNet18 fine-tuned for fresh/rotten classification
Accuracy: ~99.4% on training data
Classes: ['fresh', 'rotten']  (index 0 = fresh, index 1 = rotten)
"""

import os
import logging
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# GLOBAL MODEL STATE
# ══════════════════════════════════════════════════════════════════

_model        = None
_device       = None
_transform    = None
_class_names  = None   # e.g. ['fresh', 'rotten']
_model_loaded = False
_model_accuracy = None


def load_model(model_path: str = "freshness_model.pth") -> bool:
    """
    Load the trained ResNet18 freshness classification model.

    Returns True on success, False on failure (app falls back to mock mode).
    """
    global _model, _device, _transform, _class_names, _model_loaded, _model_accuracy

    if _model_loaded:
        return True

    if not os.path.exists(model_path):
        logger.warning(f"Model file '{model_path}' not found. Using mock predictions.")
        return False

    try:
        import torch
        import torch.nn as nn
        from torchvision import transforms, models

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {_device}")

        # ── Load checkpoint ──────────────────────────────────────
        checkpoint = torch.load(model_path, map_location=_device, weights_only=False)

        # ── Rebuild exact architecture used during training ───────
        _model = models.resnet18(weights=None)
        _model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(_model.fc.in_features, 2)
        )

        _model.load_state_dict(checkpoint["model_state_dict"])
        _model = _model.to(_device)
        _model.eval()

        # ── Metadata from checkpoint ──────────────────────────────
        _class_names    = checkpoint.get("class_names", ["fresh", "rotten"])
        _model_accuracy = checkpoint.get("accuracy", None)

        # ── Same normalisation used during training ───────────────
        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        _model_loaded = True
        acc_str = f" ({_model_accuracy:.2f}% accuracy)" if _model_accuracy else ""
        logger.info(f"Freshness model loaded successfully{acc_str}!")
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def is_model_loaded() -> bool:
    """Return True if the real trained model is loaded."""
    return _model_loaded


def get_model_accuracy() -> float | None:
    """Return the training accuracy stored in the checkpoint, or None."""
    return _model_accuracy


def get_class_names() -> list:
    """Return class names from the checkpoint."""
    return _class_names if _class_names else ["fresh", "rotten"]


# ══════════════════════════════════════════════════════════════════
# REAL ML PREDICTION
# ══════════════════════════════════════════════════════════════════

def predict_freshness(pil_image: Image.Image) -> tuple[str, float]:
    """
    Run the trained ResNet18 model on a PIL image.

    Returns:
        (prediction, confidence)
        prediction  : "fresh" or "rotten"
        confidence  : 0–100 (softmax probability × 100)
    """
    import torch

    if not _model_loaded:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Preprocess
    img_tensor = _transform(pil_image).unsqueeze(0).to(_device)

    # Forward pass
    with torch.no_grad():
        logits  = _model(img_tensor)
        probs   = torch.nn.functional.softmax(logits, dim=1)
        conf, predicted = torch.max(probs, 1)

    label      = _class_names[predicted.item()]   # "fresh" or "rotten"
    confidence = round(conf.item() * 100, 2)

    return label.lower(), confidence


# ══════════════════════════════════════════════════════════════════
# MOCK PREDICTION (fallback when model file missing)
# ══════════════════════════════════════════════════════════════════

def mock_predict(pil_image: Image.Image) -> tuple[str, float]:
    """Color-heuristic fallback — used only when model file is absent."""
    img   = np.array(pil_image)
    avg_r = img[:, :, 0].mean()
    avg_g = img[:, :, 1].mean()
    avg_b = img[:, :, 2].mean()
    brightness   = (avg_r + avg_g + avg_b) / 3
    green_ratio  = avg_g / (avg_r + avg_g + avg_b + 1e-6)
    is_dark      = brightness < 80

    if green_ratio > 0.36 and not is_dark:
        return "fresh",  round(65 + np.random.uniform(0, 25), 2)
    elif is_dark or (avg_r > avg_g and brightness < 120):
        return "rotten", round(55 + np.random.uniform(0, 30), 2)
    else:
        return "fresh",  round(50 + np.random.uniform(0, 20), 2)


# ══════════════════════════════════════════════════════════════════
# OPENCV FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_cv_features(pil_image: Image.Image) -> dict:
    """
    Extract computer-vision features (saturation, rot %, wilting %, edge density).
    Falls back to sensible defaults if OpenCV is unavailable.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available — using default CV features.")
        return {"saturation": 120, "brightness": 130,
                "wilting_pct": 10, "rot_pct": 5, "edge_density": 8}

    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    saturation = float(hsv[:, :, 1].mean())
    brightness = float(hsv[:, :, 2].mean())

    # Wilting: low-saturation pixels
    wilting_mask = hsv[:, :, 1] < 40
    wilting_pct  = round(float(wilting_mask.mean()) * 100, 2)

    # Rot: very dark pixels
    dark_mask = cv2.inRange(hsv,
                            np.array([0, 0, 0]),
                            np.array([180, 255, 60]))
    rot_pct   = round((dark_mask.sum() / 255) / dark_mask.size * 100, 2)

    # Edge density: wrinkle / texture indicator
    gray          = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges         = cv2.Canny(gray, 50, 150)
    edge_density  = round(float(edges.mean()), 2)

    return {
        "saturation":  round(saturation, 2),
        "brightness":  round(brightness, 2),
        "wilting_pct": wilting_pct,
        "rot_pct":     rot_pct,
        "edge_density": edge_density,
    }


# ══════════════════════════════════════════════════════════════════
# UNIFIED ANALYSIS ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def analyze_image(pil_image: Image.Image) -> dict:
    """
    Full pipeline: real ML prediction + CV feature extraction.

    Returns dict with:
        prediction   : "fresh" | "rotten"
        confidence   : 0–100 float
        cv_features  : dict of OpenCV metrics
        is_mock      : True only when model file was not found
        model_accuracy: training accuracy from checkpoint (or None)
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    cv_features = extract_cv_features(pil_image)

    if _model_loaded:
        prediction, confidence = predict_freshness(pil_image)
        is_mock = False
    else:
        prediction, confidence = mock_predict(pil_image)
        is_mock = True

    return {
        "prediction":     prediction,
        "confidence":     confidence,
        "cv_features":    cv_features,
        "is_mock":        is_mock,
        "model_accuracy": get_model_accuracy(),
    }
