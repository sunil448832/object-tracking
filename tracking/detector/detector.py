"""
Vehicle detector — RT-DETR v2 (ResNet-18 backbone).

Detects COCO vehicle classes (car, motorcycle, bus, truck) and pedestrians/bicycles,
returning bboxes + confidences + class labels per frame.

License: RT-DETR weights (PekingU) are Apache 2.0. Safe for commercial use.

Standalone test:
    python -m tracking.detector <image_or_video_path> [--frame N] [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor


# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME    = "PekingU/rtdetr_v2_r50vd"
DEFAULT_THR   = 0.7
# COCO class IDs we care about at a traffic junction
VEHICLE_CLASS_IDS: set[int] = {
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
}
PERSON_CLASS_ID: int = 0


@dataclass
class Detection:
    """A single detection in image coordinates."""
    xyxy: tuple[float, float, float, float]   # (x1, y1, x2, y2)
    score: float
    class_id: int
    class_name: str

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.xyxy
        return x1, y1, x2 - x1, y2 - y1


class VehicleDetector:
    """Wraps RT-DETR-v2 for per-frame vehicle detection."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        threshold: float = DEFAULT_THR,
        device: str | None = None,
        class_ids: Iterable[int] | None = None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.class_ids = set(class_ids) if class_ids is not None else VEHICLE_CLASS_IDS
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.use_fp16 = self.device.type == "cuda"
        print(f"[Detector] Loading {model_name} on {self.device} "
              f"(fp16={self.use_fp16}) ...")
        self.processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_name)
        self.model.eval().to(self.device)
        if self.use_fp16:
            self.model = self.model.half()
        self.id2label = self.model.config.id2label
        print(f"[Detector] Loaded. Filtering for class IDs: "
              f"{sorted(self.class_ids)} "
              f"({[self.id2label[c] for c in sorted(self.class_ids)]})")

    # ── Public API ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        """
        Run detection on one frame.

        Args:
            frame_rgb: HxWx3 uint8 numpy array, RGB.

        Returns:
            List of Detection objects, filtered to the configured class IDs
            and above self.threshold.
        """
        image = Image.fromarray(frame_rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        if self.use_fp16:
            inputs = {k: (v.half() if torch.is_floating_point(v) else v)
                      for k, v in inputs.items()}
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)  # (H, W)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
        )[0]

        dets: list[Detection] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            cid = int(label.item())
            if cid not in self.class_ids:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            dets.append(Detection(
                xyxy=(x1, y1, x2, y2),
                score=float(score.item()),
                class_id=cid,
                class_name=self.id2label[cid],
            ))
        return dets


# ── Utilities for standalone test ────────────────────────────────────────────
def _load_frame(path: str, frame_index: int = 0) -> np.ndarray:
    """Load an image or a single frame from a video (returns RGB np.ndarray)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Assume video
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _annotate(frame_rgb: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Draw bounding boxes with matplotlib-style colors on a BGR copy for saving."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det.xyxy]
        color = (0, 200, 50)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.score:.0%}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(bgr, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, cv2.FILLED)
        cv2.putText(bgr, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


# ── Standalone test ──────────────────────────────────────────────────────────
def _test() -> None:
    ap = argparse.ArgumentParser(description="RT-DETR vehicle detector — standalone test")
    ap.add_argument("input", help="Path to image or video")
    ap.add_argument("--frame", type=int, default=0, help="Frame index (video only)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THR, help="Confidence threshold")
    ap.add_argument("--output", default="detector_test_output.jpg", help="Annotated output path")
    args = ap.parse_args()

    frame = _load_frame(args.input, args.frame)
    print(f"[TEST] Loaded frame  shape={frame.shape}  from={args.input}")

    detector = VehicleDetector(threshold=args.threshold)
    dets = detector.detect(frame)
    print(f"\n[TEST] Detections ({len(dets)}):")
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = [int(v) for v in d.xyxy]
        print(f"  [{i:2}] {d.class_name:<12} {d.score:.2%}   "
              f"bbox=({x1},{y1},{x2},{y2})  size=({x2-x1}x{y2-y1})")

    out_bgr = _annotate(frame, dets)
    cv2.imwrite(args.output, out_bgr)
    print(f"\n[TEST] Annotated frame saved → {args.output}")


if __name__ == "__main__":
    _test()
