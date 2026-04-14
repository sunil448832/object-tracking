"""
License plate OCR — fast-plate-ocr (CCT, ONNX).

Given a plate crop (or batch of crops), returns the recognized plate string and
optional region/country prediction.

License: fast-plate-ocr is Apache 2.0. Safe for commercial use.

Standalone test:
    python -m tracking.plate_ocr <plate_crop.jpg> [crop2.jpg ...]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer


# ── Config ───────────────────────────────────────────────────────────────────
# Available:
#   cct-xs-v1-global-model   (smallest, fastest, text only)
#   cct-xs-v2-global-model   (tiny, text + region)
#   cct-s-v1-global-model    (larger, text only)
#   cct-s-v2-global-model    (larger, text + region)
DEFAULT_MODEL = "cct-s-v2-global-model"


@dataclass
class PlateReading:
    """OCR result for a single plate crop."""
    text: str
    region: str | None = None
    char_probs: np.ndarray | None = None  # per-character probs if requested


class PlateOCR:
    """Wraps fast-plate-ocr's LicensePlateRecognizer."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        print(f"[PlateOCR] Loading {model_name} (ONNX) ...")
        self._rec = LicensePlateRecognizer(model_name)
        print("[PlateOCR] Loaded.")

    # ── Public API ────────────────────────────────────────────────────────────
    def read(
        self,
        crops: Sequence[np.ndarray] | np.ndarray,
        return_confidence: bool = False,
    ) -> list[PlateReading]:
        """
        Run OCR on one or more plate crops.

        Args:
            crops: a single HxWx3 RGB uint8 array, a list of such arrays,
                   or a 4D (N, H, W, 3) RGB uint8 batch. All inputs must be RGB.
            return_confidence: if True, populate PlateReading.char_probs.

        Returns:
            List of PlateReading, one per input crop.
        """
        # Normalize to a list of np.ndarray for the underlying API
        if isinstance(crops, np.ndarray) and crops.ndim == 3:
            inputs: list[np.ndarray] = [crops]
        elif isinstance(crops, np.ndarray) and crops.ndim == 4:
            inputs = [crops[i] for i in range(crops.shape[0])]
        else:
            inputs = list(crops)

        if not inputs:
            return []

        preds = self._rec.run(inputs, return_confidence=return_confidence)

        readings: list[PlateReading] = []
        for p in preds:
            readings.append(PlateReading(
                text=p.plate,
                region=getattr(p, "region", None),
                char_probs=getattr(p, "char_probs", None) if return_confidence else None,
            ))
        return readings


# ── Standalone test ──────────────────────────────────────────────────────────
def _test() -> None:
    ap = argparse.ArgumentParser(description="fast-plate-ocr recognizer — standalone test")
    ap.add_argument("crops", nargs="+", help="Paths to plate crop images")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OCR model variant")
    ap.add_argument("--confidence", action="store_true", help="Also print per-character confidence")
    args = ap.parse_args()

    imgs: list[np.ndarray] = []
    for p in args.crops:
        if not Path(p).exists():
            raise FileNotFoundError(p)
        bgr = cv2.imread(p)
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"[TEST] Loaded {p}  shape={rgb.shape}")
        imgs.append(rgb)

    ocr = PlateOCR(model_name=args.model)
    readings = ocr.read(imgs, return_confidence=args.confidence)

    print(f"\n[TEST] Results ({len(readings)}):")
    for i, (path, r) in enumerate(zip(args.crops, readings), 1):
        print(f"  [{i}] {Path(path).name}")
        print(f"       text   : {r.text!r}")
        if r.region:
            print(f"       region : {r.region}")
        if args.confidence and r.char_probs is not None:
            probs = r.char_probs
            maxp = probs.max(axis=-1) if probs.ndim == 2 else probs
            print(f"       per-char max prob : {np.round(maxp, 3).tolist()}")


if __name__ == "__main__":
    _test()
