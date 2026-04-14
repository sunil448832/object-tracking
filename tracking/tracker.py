"""
Thin wrapper around vendored BoT-SORT.

Accepts detections from any detector and pre-computed appearance embeddings
from any re-ID model (default: DINOv2 from tracking.reid).

License: wraps MIT-licensed BoT-SORT (NirAharon/BoT-SORT) vendored under
tracking/botsort/. See tracking/botsort/LICENSE.

Standalone test:
    python -m tracking.tracker <video_path> [--frames N] [--stride K]

The standalone test runs the full detector → DINOv2 → BoT-SORT chain on
the first N frames of a video and writes an annotated output video.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from .botsort.bot_sort import BoTSORT


# ── Config ───────────────────────────────────────────────────────────────────
@dataclass
class BoTSORTConfig:
    """Hyperparameters for the BoT-SORT tracker.

    Mirrors the CLI args that BoT-SORT expects via argparse, packaged
    as a dataclass for programmatic use.
    """

    # Detection confidence thresholds
    track_high_thresh: float = 0.5      # trust these for new tracks + first assoc
    track_low_thresh: float  = 0.1      # below this → discarded
    new_track_thresh: float  = 0.6      # spawn a new track only above this
    match_thresh: float      = 0.8      # IoU association threshold (round 1)

    # Lifecycle
    track_buffer: int = 30              # frames to keep a lost track before removing

    # Re-ID / appearance
    with_reid: bool = True              # use appearance features in association
    proximity_thresh: float = 0.5       # gate: IoU must be within this for appearance to count
    appearance_thresh: float = 0.25     # cosine-distance threshold for appearance

    # MOT20 mode turns off score-fusion (for crowded scenes). Default off.
    mot20: bool = False

    # Camera motion compensation. "none" disables — correct for fixed cameras.
    cmc_method: str = "none"

    # Cosmetic
    name: str = "botsort"
    ablation: bool = False

    def to_ns(self) -> SimpleNamespace:
        return SimpleNamespace(**self.__dict__)


# ── Public result types ──────────────────────────────────────────────────────
@dataclass
class TrackedObject:
    """A single tracked object in the current frame."""
    track_id: int
    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str = ""

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.xyxy
        return x1, y1, x2 - x1, y2 - y1


# ── Tracker ──────────────────────────────────────────────────────────────────
class Tracker:
    """Wraps BoTSORT with a simpler API keyed off Detection objects + features."""

    def __init__(self, config: BoTSORTConfig | None = None, frame_rate: int = 30) -> None:
        self.config = config or BoTSORTConfig()
        self._botsort = BoTSORT(self.config.to_ns(), frame_rate=frame_rate)
        self._id2name: dict[int, str] = {}
        print(f"[Tracker] BoT-SORT initialised  "
              f"(with_reid={self.config.with_reid}, cmc={self.config.cmc_method})")

    def update(
        self,
        detections: list,                 # list[detector.Detection]-like (has .xyxy, .score, .class_id, .class_name)
        features: np.ndarray | None,      # (N, D) L2-normalised embeddings, aligned with detections
        frame_bgr: np.ndarray | None = None,
    ) -> list[TrackedObject]:
        """Advance the tracker by one frame. Returns currently active tracks."""
        if len(detections) == 0:
            output_results = np.empty((0, 6), dtype=np.float32)
            features_arr = np.empty((0, 0), dtype=np.float32) if features is None else features
        else:
            output_results = np.asarray(
                [(*d.xyxy, float(d.score), float(d.class_id)) for d in detections],
                dtype=np.float32,
            )
            for d in detections:
                self._id2name.setdefault(int(d.class_id), getattr(d, "class_name", str(d.class_id)))
            features_arr = features

        if self.config.with_reid and (features is None or len(detections) == 0):
            # BoT-SORT expects aligned features when with_reid=True.
            # If we have zero detections it doesn't matter; otherwise this is a caller bug.
            if len(detections) > 0 and features is None:
                raise ValueError("with_reid=True but features is None")

        online_targets = self._botsort.update(
            output_results,
            frame_bgr if frame_bgr is not None else np.zeros((1, 1, 3), dtype=np.uint8),
            features=features_arr,
        )

        out: list[TrackedObject] = []
        for t in online_targets:
            x, y, w, h = t.tlwh
            cid = int(getattr(t, "cls", -1)) if hasattr(t, "cls") else -1
            out.append(TrackedObject(
                track_id=int(t.track_id),
                xyxy=(float(x), float(y), float(x + w), float(y + h)),
                score=float(t.score),
                class_id=cid,
                class_name=self._id2name.get(cid, ""),
            ))
        return out


# ── Standalone test: video → detector → DINOv2 → tracker → annotated video ──
def _color_for(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color per track id."""
    rng = np.random.default_rng(track_id * 9973 + 1)
    return tuple(int(v) for v in rng.integers(40, 255, size=3))


def _draw(frame_bgr: np.ndarray, tracks: list[TrackedObject]) -> np.ndarray:
    out = frame_bgr.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.xyxy]
        color = _color_for(t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{t.track_id} {t.class_name}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - bl - 6), (x1 + tw + 4, y1), color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _test() -> None:
    ap = argparse.ArgumentParser(description="BoT-SORT wrapper — standalone test")
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("--frames", type=int, default=60, help="Max frames to process")
    ap.add_argument("--stride", type=int, default=1, help="Process every Kth frame")
    ap.add_argument("--threshold", type=float, default=0.4, help="Detector confidence threshold")
    ap.add_argument("--output", default="tracker_test_output.mp4", help="Annotated output video path")
    args = ap.parse_args()

    # Lazy-import model modules so unit-testing the tracker alone doesn't require them.
    from .detector import VehicleDetector
    from .reid import DinoV2Embedder

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[TEST] Video: {args.video}  {w}x{h}  {fps:.2f} fps  frames={total}")

    detector = VehicleDetector(threshold=args.threshold)
    embedder = DinoV2Embedder()
    tracker = Tracker(BoTSORTConfig(), frame_rate=int(round(fps)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps / max(1, args.stride), (w, h))

    processed = 0
    frame_idx = -1
    while processed < args.frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.stride != 0:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets = detector.detect(frame_rgb)

        feats = None
        if len(dets) > 0 and tracker.config.with_reid:
            crops = []
            for d in dets:
                x1, y1, x2, y2 = [max(0, int(v)) for v in d.xyxy]
                x2 = min(w, x2); y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    # Degenerate box — fall back to a 1-pixel crop (rare)
                    crops.append(Image.new("RGB", (14, 14)))
                else:
                    crops.append(Image.fromarray(frame_rgb[y1:y2, x1:x2]))
            feats = embedder.embed(crops)

        tracks = tracker.update(dets, feats, frame_bgr=frame_bgr)
        annotated = _draw(frame_bgr, tracks)
        writer.write(annotated)

        processed += 1
        if processed % 5 == 0 or processed == 1:
            print(f"[TEST] frame {frame_idx:4d}  "
                  f"dets={len(dets):2d}  tracks={len(tracks):2d}  "
                  f"ids={sorted(t.track_id for t in tracks)}")

    cap.release()
    writer.release()
    print(f"\n[TEST] Processed {processed} frames → {args.output}")


if __name__ == "__main__":
    _test()
