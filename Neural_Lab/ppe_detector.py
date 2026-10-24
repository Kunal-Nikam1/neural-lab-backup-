"""
PPE Compliance Detection Engine
================================
Two-stage pipeline for hi-vis vest detection:
  Stage 1: Person detection via MediaPipe EfficientDet-Lite0
  Stage 2: Vest classification via CLIP zero-shot (no training data needed)

Author: Neural_Lab
"""

import os
import sys
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ---------------------------------------------------------------------------
#  Stage 1 — Person Detection (MediaPipe EfficientDet-Lite0)
# ---------------------------------------------------------------------------

class PersonDetector:
    """Detects persons in a frame using MediaPipe's EfficientDet-Lite0.

    Automatically downloads the TFLite model on first use (~4 MB).
    Returns bounding boxes as (x, y, w, h, confidence) tuples.
    """

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
    )

    def __init__(self, score_threshold: float = 0.45, model_dir: str = None):
        """
        Args:
            score_threshold: Minimum confidence to keep a person detection.
            model_dir: Directory to store/find the tflite model.
                       Defaults to the same directory as this script.
        """
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        self._model_path = os.path.join(model_dir, "efficientdet_lite0.tflite")
        self._ensure_model()

        base_options = mp_python.BaseOptions(
            model_asset_path=self._model_path
        )
        options = mp_vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            score_threshold=score_threshold,
            category_allowlist=["person"],
        )
        self.detector = mp_vision.ObjectDetector.create_from_options(options)
        print(f"[PersonDetector] Ready  (threshold={score_threshold})")

    # -- internal ----------------------------------------------------------

    def _ensure_model(self):
        """Download the TFLite model if it doesn't exist locally."""
        if os.path.exists(self._model_path):
            return
        print(f"[PersonDetector] Downloading EfficientDet-Lite0 model …")
        urllib.request.urlretrieve(self.MODEL_URL, self._model_path)
        print(f"[PersonDetector] Saved to {self._model_path}")

    # -- public API --------------------------------------------------------

    def detect(self, frame: np.ndarray):
        """Detect persons in a BGR OpenCV frame.

        Args:
            frame: BGR numpy array from cv2.

        Returns:
            List of tuples (x, y, w, h, confidence) for each detected person.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        detections = []
        for det in result.detections:
            bbox = det.bounding_box
            conf = det.categories[0].score
            detections.append((
                bbox.origin_x,
                bbox.origin_y,
                bbox.width,
                bbox.height,
                conf,
            ))
        return detections


# ---------------------------------------------------------------------------
#  Stage 2 — Vest Classification (CLIP zero-shot)
# ---------------------------------------------------------------------------

class VestClassifier:
    """Zero-shot hi-vis vest classifier using OpenAI CLIP.

    No training or fine-tuning needed — the model compares the cropped
    person image against text descriptions of "wearing vest" vs "not
    wearing vest" and picks the best match.
    """

    # Carefully engineered prompts for hi-vis vest detection
    PROMPTS = [
        "a person wearing a bright fluorescent high-visibility safety vest "
        "with reflective strips",
        "a person wearing normal everyday clothes without any safety vest",
    ]

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        """
        Args:
            model_id: HuggingFace model ID for CLIP.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VestClassifier] Loading CLIP on {self.device.upper()} …")

        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

        print(f"[VestClassifier] Ready  (model={model_id})")

    # -- public API --------------------------------------------------------

    def classify(self, person_crop_bgr: np.ndarray):
        """Classify whether a person crop shows a hi-vis vest.

        Args:
            person_crop_bgr: BGR numpy array — cropped region of one person.

        Returns:
            Tuple of (is_compliant: bool, confidence: float).
            is_compliant is True when the person is wearing a vest.
            confidence is the softmax probability of the winning class.
        """
        # Guard: skip tiny crops that would confuse CLIP
        if person_crop_bgr.shape[0] < 20 or person_crop_bgr.shape[1] < 20:
            return False, 0.0

        rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        inputs = self.processor(
            text=self.PROMPTS,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

        vest_prob = float(probs[0])
        no_vest_prob = float(probs[1])

        is_compliant = vest_prob > no_vest_prob
        confidence = vest_prob if is_compliant else no_vest_prob

        return is_compliant, confidence


# ---------------------------------------------------------------------------
#  Combined Pipeline
# ---------------------------------------------------------------------------

class PPECompliancePipeline:
    """End-to-end PPE compliance pipeline combining both stages.

    Usage:
        pipeline = PPECompliancePipeline()
        results  = pipeline.process(frame)
        # results is a list of dicts with keys:
        #   x, y, w, h, is_compliant, confidence
    """

    def __init__(
        self,
        person_threshold: float = 0.45,
        crop_padding: int = 15,
    ):
        self.person_detector = PersonDetector(score_threshold=person_threshold)
        self.vest_classifier = VestClassifier()
        self.crop_padding = crop_padding

    def process(self, frame: np.ndarray):
        """Run the full two-stage pipeline on a BGR frame.

        Returns:
            List of dicts, one per detected person:
              {
                "x": int, "y": int, "w": int, "h": int,
                "is_compliant": bool,
                "confidence": float,
              }
        """
        h, w = frame.shape[:2]
        pad = self.crop_padding

        # Stage 1 — find all persons
        persons = self.person_detector.detect(frame)

        results = []
        for (px, py, pw, ph, _pconf) in persons:
            # Compute padded crop region
            x1 = max(0, px - pad)
            y1 = max(0, py - pad)
            x2 = min(w, px + pw + pad)
            y2 = min(h, py + ph + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Stage 2 — classify vest
            is_compliant, conf = self.vest_classifier.classify(crop)

            results.append({
                "x": px, "y": py, "w": pw, "h": ph,
                "is_compliant": is_compliant,
                "confidence": conf,
            })

        return results


# ---------------------------------------------------------------------------
#  Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("PPE Detector module loaded. Import this in ppe_live.py.")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  Device:         {'GPU' if torch.cuda.is_available() else 'CPU'}")
