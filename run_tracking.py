"""
Full ANPR tracking pipeline.

Per frame:
  RT-DETR (vehicle detection)
    → crop each vehicle
      → DINOv2 (appearance embedding, batched)
      → YOLOv9 (plate detection inside each crop)
        → fast-plate-ocr (plate text)
  → BoT-SORT (tracker: motion + appearance)
  → Per-track plate memory (keep the best-confidence reading seen so far)
  → Annotated video output

Usage:
    python run_tracking.py <video_path> [options]

Example:
    python run_tracking.py data/images/NVR-ABD6_ch50_main_20260408101835_20260408101854.mp4 \
        --output data/results/pipeline.mp4 --frames 200 --threshold 0.5

All components commercial-safe (Apache 2.0 / MIT):
  Detector : PekingU/rtdetr_v2_r18vd      (transformers)
  Re-ID    : facebookresearch/dinov2_vits14_reg
  Tracker  : NirAharon/BoT-SORT            (vendored, MIT)
  Plate det: open-image-models YOLOv9
  Plate OCR: fast-plate-ocr                (CCT)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tracking.detector import VehicleDetector
from tracking.reid import DinoV2Embedder
from tracking.tracker import Tracker, BoTSORTConfig, TrackedObject
from tracking.plate_detector import PlateDetector
from tracking.plate_ocr import PlateOCR


# ── Per-track plate memory ───────────────────────────────────────────────────
@dataclass
class PlateMemory:
    """Best plate reading we've accumulated for a single track."""
    text: str = ""
    region: str | None = None
    confidence: float = 0.0      # mean per-char prob of the best reading
    det_score: float = 0.0       # YOLOv9 plate-detection confidence
    seen_frames: int = 0         # how many times we've OCR'd this track


def _mean_char_confidence(char_probs: np.ndarray | None) -> float:
    if char_probs is None:
        return 0.0
    p = np.asarray(char_probs)
    if p.ndim == 2:
        p = p.max(axis=-1)
    return float(np.mean(p)) if p.size else 0.0


# ── Drawing ──────────────────────────────────────────────────────────────────
def _color_for(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 9973 + 1)
    return tuple(int(v) for v in rng.integers(40, 255, size=3))


def _annotate_frame(
    frame_bgr: np.ndarray,
    tracks: list[TrackedObject],
    plate_memory: dict[int, PlateMemory],
    plate_boxes_this_frame: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    out = frame_bgr.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.xyxy]
        color = _color_for(t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        mem = plate_memory.get(t.track_id)
        plate_text = mem.text if mem and mem.text else ""
        label = f"#{t.track_id} {t.class_name}"
        if plate_text:
            label += f" | {plate_text}"

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, y1 - th - bl - 6), (x1 + tw + 4, y1), color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw plate bbox if we detected one this frame for this track
        pb = plate_boxes_this_frame.get(t.track_id)
        if pb is not None:
            px1, py1, px2, py2 = pb
            cv2.rectangle(out, (px1, py1), (px2, py2), (0, 0, 255), 2)
    return out


# ── Pipeline ─────────────────────────────────────────────────────────────────
def run_pipeline(
    video_path: str,
    output_path: str,
    max_frames: int = 0,
    stride: int = 1,
    det_threshold: float = 0.7,
    plate_conf_improve_only: bool = True,
    ocr_every_n_frames_per_track: int = 3,
    min_vehicle_area: int = 2000,       # skip OCR on tiny (distant) vehicles
    min_ocr_confidence: float = 0.5,    # reject OCR reads below this mean char prob
    lock_in_confidence: float = 0.95,   # once a track hits this, stop OCR'ing it
    plate_detector_model: str = "yolo-v9-s-608-license-plate-end2end",
    plate_ocr_model: str = "cct-s-v2-global-model",
) -> dict[int, PlateMemory]:
    """Run the full pipeline on a video. Returns track_id → best PlateMemory."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Pipeline] Video: {video_path}")
    print(f"[Pipeline] {W}x{H}  fps={fps:.2f}  frames={total}")

    # Load all models
    detector = VehicleDetector(threshold=det_threshold)
    embedder = DinoV2Embedder()
    plate_det = PlateDetector(model_name=plate_detector_model, threshold=0.3)
    plate_ocr = PlateOCR(model_name=plate_ocr_model)
    tracker = Tracker(BoTSORTConfig(), frame_rate=int(round(fps)))

    # Video writer at stride-adjusted fps
    out_fps = fps / max(1, stride)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (W, H))

    # Per-track state
    plate_memory: dict[int, PlateMemory] = {}
    last_ocr_frame: dict[int, int] = {}

    processed = 0
    frame_idx = -1
    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        if max_frames and processed >= max_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1) Detect vehicles
        dets = detector.detect(frame_rgb)

        # 2) Appearance embeddings (batched)
        feats = None
        crops_rgb: list[np.ndarray] = []
        if len(dets):
            for d in dets:
                x1, y1, x2, y2 = [max(0, int(v)) for v in d.xyxy]
                x2 = min(W, x2); y2 = min(H, y2)
                if x2 > x1 and y2 > y1:
                    crops_rgb.append(frame_rgb[y1:y2, x1:x2])
                else:
                    crops_rgb.append(np.zeros((14, 14, 3), dtype=np.uint8))
            feats = embedder.embed([Image.fromarray(c) for c in crops_rgb])

        # 3) Track
        tracks = tracker.update(dets, feats, frame_bgr=frame_bgr)

        # 4) Plate detection + OCR per tracked vehicle — BATCHED across tracks.
        plate_boxes_this_frame: dict[int, tuple[int, int, int, int]] = {}

        # 4a) Select which tracks need OCR work this frame
        ocr_track_ids: list[int] = []
        ocr_offsets: list[tuple[int, int]] = []           # (vx1, vy1) of each crop in frame coords
        ocr_vehicle_crops: list[np.ndarray] = []          # RGB crops aligned with ocr_track_ids
        for t in tracks:
            # Rate limit
            last = last_ocr_frame.get(t.track_id, -10**9)
            if frame_idx - last < ocr_every_n_frames_per_track:
                continue
            # Lock-in: stop OCR'ing tracks we've already read with high confidence
            mem = plate_memory.get(t.track_id)
            if mem is not None and mem.confidence >= lock_in_confidence:
                continue

            vx1, vy1, vx2, vy2 = [max(0, int(v)) for v in t.xyxy]
            vx2 = min(W, vx2); vy2 = min(H, vy2)
            area = (vx2 - vx1) * (vy2 - vy1)
            # Skip tiny / degenerate boxes
            if vx2 <= vx1 or vy2 <= vy1 or area < min_vehicle_area:
                continue

            ocr_track_ids.append(t.track_id)
            ocr_offsets.append((vx1, vy1))
            ocr_vehicle_crops.append(frame_rgb[vy1:vy2, vx1:vx2])

        # 4b) Batched plate detection on all vehicle crops
        batch_plate_dets: list[list] = (
            plate_det.detect_batch(ocr_vehicle_crops) if ocr_vehicle_crops else []
        )

        # 4c) Collect plate crops that were actually detected → batched OCR
        plate_crops_rgb: list[np.ndarray] = []
        plate_owner_ids: list[int] = []                   # track_id per plate crop
        plate_det_scores: list[float] = []
        for i, (tid, offset, pdets) in enumerate(zip(ocr_track_ids, ocr_offsets, batch_plate_dets)):
            last_ocr_frame[tid] = frame_idx               # record attempt either way
            if not pdets:
                continue
            pd = max(pdets, key=lambda p: p.score)
            px1, py1, px2, py2 = pd.xyxy
            vcrop = ocr_vehicle_crops[i]                  # RGB crop already cut earlier
            plate_crop = vcrop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue
            plate_crops_rgb.append(plate_crop)
            plate_owner_ids.append(tid)
            plate_det_scores.append(pd.score)
            plate_boxes_this_frame[tid] = (
                offset[0] + px1, offset[1] + py1,
                offset[0] + px2, offset[1] + py2,
            )

        # 4d) Single batched OCR call for all plate crops in this frame
        if plate_crops_rgb:
            readings = plate_ocr.read(plate_crops_rgb, return_confidence=True)
            for tid, det_score, r in zip(plate_owner_ids, plate_det_scores, readings):
                conf = _mean_char_confidence(r.char_probs)
                # Reject low-confidence / empty OCR reads
                if conf < min_ocr_confidence or not r.text:
                    mem = plate_memory.get(tid)
                    if mem is not None:
                        mem.seen_frames += 1
                    continue
                mem = plate_memory.get(tid)
                accept = (
                    mem is None
                    or not plate_conf_improve_only
                    or (conf > mem.confidence + 0.02)      # small hysteresis
                )
                if accept:
                    plate_memory[tid] = PlateMemory(
                        text=r.text,
                        region=r.region,
                        confidence=conf,
                        det_score=det_score,
                        seen_frames=(mem.seen_frames if mem else 0) + 1,
                    )
                elif mem is not None:
                    mem.seen_frames += 1

        # 5) Annotate + write
        annotated = _annotate_frame(frame_bgr, tracks, plate_memory, plate_boxes_this_frame)
        writer.write(annotated)

        processed += 1
        if processed % 5 == 0 or processed == 1:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(f"[Pipeline] frame {frame_idx:4d}  "
                  f"dets={len(dets):2d}  tracks={len(tracks):2d}  "
                  f"plates_known={sum(1 for m in plate_memory.values() if m.text)}  "
                  f"({rate:.2f} fps)")

    cap.release()
    writer.release()

    total_elapsed = time.time() - t0
    print(f"\n[Pipeline] Processed {processed} frames in {total_elapsed:.1f}s "
          f"({processed / max(total_elapsed, 1e-6):.2f} fps avg)")
    print(f"[Pipeline] Output video: {output_path}")
    print(f"\n[Pipeline] Final plate readings ({len(plate_memory)} tracks with plates):")
    for tid in sorted(plate_memory):
        m = plate_memory[tid]
        if not m.text:
            continue
        print(f"  track #{tid:3d}  plate={m.text!r}  "
              f"region={m.region or '-'}  "
              f"ocr_conf={m.confidence:.2f}  seen={m.seen_frames}")

    return plate_memory


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Full ANPR tracking pipeline")
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--output", default="pipeline_output.mp4", help="Annotated output video path")
    ap.add_argument("--frames", type=int, default=0, help="Max frames to process (0 = all)")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--threshold", type=float, default=0.7, help="Vehicle detector confidence threshold")
    ap.add_argument("--ocr-every", type=int, default=3,
                    help="Min frames between OCR calls per track")
    ap.add_argument("--min-vehicle-area", type=int, default=2000,
                    help="Skip OCR for vehicles with bbox area (px²) below this")
    ap.add_argument("--min-ocr-conf", type=float, default=0.5,
                    help="Reject OCR reads with mean per-char confidence below this")
    ap.add_argument("--lock-in-conf", type=float, default=0.95,
                    help="Once a track reaches this OCR confidence, stop re-OCR'ing it")
    ap.add_argument("--plate-model", default="yolo-v9-s-608-license-plate-end2end")
    ap.add_argument("--ocr-model", default="cct-s-v2-global-model")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.frames,
        stride=args.stride,
        det_threshold=args.threshold,
        ocr_every_n_frames_per_track=args.ocr_every,
        min_vehicle_area=args.min_vehicle_area,
        min_ocr_confidence=args.min_ocr_conf,
        lock_in_confidence=args.lock_in_conf,
        plate_detector_model=args.plate_model,
        plate_ocr_model=args.ocr_model,
    )


if __name__ == "__main__":
    main()
