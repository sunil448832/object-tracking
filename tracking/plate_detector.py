"""
License plate detector — YOLOv9 (open-image-models, ONNX).

Detects plate bounding boxes within an image or vehicle crop. Designed to be
run on cropped vehicle regions (output of the vehicle detector) during the
tracking pipeline.

License: open-image-models is Apache 2.0. Safe for commercial use.

Standalone test:
    python -m tracking.plate_detector <image_or_video_path> [--frame N]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from open_image_models import LicensePlateDetector as _OIMLicensePlateDetector


# ── Config ───────────────────────────────────────────────────────────────────
# Available: yolo-v9-s-608 / yolo-v9-t-{640,512,416,384,256}-license-plate-end2end
# Recommended for vehicle crops: t-384 (good speed/accuracy for typical crop sizes)
# Recommended for full 4K frames : s-608 (more accurate on tiny distant plates)
DEFAULT_MODEL = "yolo-v9-t-384-license-plate-end2end"
DEFAULT_THR   = 0.3


@dataclass
class PlateDetection:
    """A plate detection in image coordinates."""
    xyxy: tuple[int, int, int, int]   # (x1, y1, x2, y2) — integer pixel coords
    score: float
    label: str = "License Plate"

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.xyxy
        return x1, y1, x2 - x1, y2 - y1


class PlateDetector:
    """Wraps open-image-models' YOLOv9 license plate detector."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THR,
        providers: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        print(f"[PlateDet] Loading {model_name} (ONNX) ...")
        self._det = _OIMLicensePlateDetector(
            detection_model=model_name,  # type: ignore[arg-type]
            conf_thresh=threshold,
            providers=providers,  # default: ORT chooses (CPU by default if no GPU)
        )
        print(f"[PlateDet] Loaded.")

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, image_rgb: np.ndarray) -> list[PlateDetection]:
        """Run plate detection on one RGB image. Returns list of PlateDetection."""
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        raw = self._det.predict(image_bgr)
        return [self._to_plate_detection(r) for r in raw]

    def detect_batch(self, images_rgb: list[np.ndarray]) -> list[list[PlateDetection]]:
        """Run plate detection on a batch of RGB images. Images may be different sizes."""
        if not images_rgb:
            return []
        images_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_rgb]
        raw_batch = self._det.predict(images_bgr)  # list[list[DetectionResult]]
        return [[self._to_plate_detection(r) for r in raw] for raw in raw_batch]

    @staticmethod
    def _to_plate_detection(r) -> PlateDetection:
        bb = r.bounding_box
        return PlateDetection(
            xyxy=(int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)),
            score=float(r.confidence),
            label=str(r.label),
        )


# ── Utilities for standalone test ────────────────────────────────────────────
def _load_frame(path: str, frame_index: int = 0) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _annotate(frame_rgb: np.ndarray, dets: list[PlateDetection]) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for det in dets:
        x1, y1, x2, y2 = det.xyxy
        color = (0, 0, 255)   # red in BGR
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"plate {det.score:.0%}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(bgr, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, cv2.FILLED)
        cv2.putText(bgr, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


# ── Standalone test ──────────────────────────────────────────────────────────
def _test() -> None:
    ap = argparse.ArgumentParser(description="YOLOv9 license plate detector — standalone test")
    ap.add_argument("input", help="Path to image or video")
    ap.add_argument("--frame", type=int, default=0, help="Frame index (video only)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model variant")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THR, help="Confidence threshold")
    ap.add_argument("--output", default="plate_detector_test.jpg", help="Annotated output path")
    args = ap.parse_args()

    frame = _load_frame(args.input, args.frame)
    print(f"[TEST] Loaded frame  shape={frame.shape}  from={args.input}")

    det = PlateDetector(model_name=args.model, threshold=args.threshold)
    dets = det.detect(frame)
    print(f"\n[TEST] Plates detected: {len(dets)}")
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = d.xyxy
        print(f"  [{i:2}] {d.label:<13} {d.score:.2%}   "
              f"bbox=({x1},{y1},{x2},{y2})  size=({x2-x1}x{y2-y1})")

    out_bgr = _annotate(frame, dets)
    cv2.imwrite(args.output, out_bgr)
    print(f"\n[TEST] Annotated frame saved → {args.output}")


if __name__ == "__main__":
    _test()
