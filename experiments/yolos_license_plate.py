"""
License Plate Detection + OCR
  Detector : nickmuchi/yolos-small-finetuned-license-plate-detection  (HF, Apache 2.0)
  OCR      : fast-plate-ocr  cct-s-v2-global-model                     (ONNX, Apache 2.0)

Install dependencies:
    pip install transformers torch Pillow fast-plate-ocr
"""

import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import YolosImageProcessor, YolosForObjectDetection
from fast_plate_ocr import LicensePlateRecognizer


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID     = "nickmuchi/yolos-small-finetuned-license-plate-detection"
# Lightweight plate-specific OCR (Compact Convolutional Transformer, ~few MB).
# Alternatives: "cct-xs-v1-global-model" (smallest), "cct-s-v1-global-model".
OCR_MODEL_ID = "cct-s-v2-global-model"
THRESHOLD  = 0.5      # confidence threshold (0.0 – 1.0)
BOX_COLOR  = "red"
TEXT_COLOR = "white"
BOX_WIDTH  = 3
# ─────────────────────────────────────────────────────────────────────────────


def load_model(model_id: str = MODEL_ID):
    """Load the YOLOS feature extractor and model from Hugging Face."""
    print(f"Loading model: {model_id} ...")
    feature_extractor = YolosImageProcessor.from_pretrained(model_id)
    model = YolosForObjectDetection.from_pretrained(model_id)
    model.eval()
    print("Model loaded successfully.\n")
    return feature_extractor, model


def load_ocr(model_id: str = OCR_MODEL_ID) -> LicensePlateRecognizer:
    """Load the fast-plate-ocr recognizer (ONNX, plate-specific)."""
    print(f"Loading OCR model: {model_id} ...")
    recognizer = LicensePlateRecognizer(model_id)
    print("OCR model loaded successfully.\n")
    return recognizer


def read_plate_text(image: Image.Image, box: dict, recognizer: LicensePlateRecognizer) -> dict:
    """Crop the plate region from `image` and run OCR on it.

    Returns a dict with keys: text, region (may be None).
    """
    crop = image.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
    arr = np.array(crop)  # RGB uint8, (H, W, 3)
    pred = recognizer.run(arr)[0]
    return {"text": pred.plate, "region": getattr(pred, "region", None)}


def detect_plates(image_path: str,
                  feature_extractor,
                  model,
                  threshold: float = THRESHOLD):
    """
    Run license plate detection on a single image file.

    Args:
        image_path:        Path to the input image.
        feature_extractor: YOLOS feature extractor.
        model:             YOLOS detection model.
        threshold:         Minimum confidence score to keep a detection.

    Returns:
        List of dicts with keys: label, score, box (xmin, ymin, xmax, ymax).
    """
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Inference (no gradient needed)
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: convert raw outputs → bounding boxes in image coordinates
    target_sizes = torch.tensor([image.size[::-1]])   # (height, width)
    results = feature_extractor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(c, 2) for c in box.tolist()]
        detections.append({
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 4),
            "box":   {
                "xmin": box[0],
                "ymin": box[1],
                "xmax": box[2],
                "ymax": box[3],
            },
        })

    return image, detections


def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)

    # Try to load a nicer font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        box   = det["box"]
        score = det["score"]
        plate_text = det.get("text", "")

        # Draw bounding box
        draw.rectangle(
            [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
            outline=BOX_COLOR,
            width=BOX_WIDTH,
        )

        # Label shows the plate text if available, else falls back to class label
        header = plate_text if plate_text else det["label"]
        text = f"{header} ({score:.0%})"
        text_bbox = draw.textbbox((box["xmin"], box["ymin"]), text, font=font)
        draw.rectangle(text_bbox, fill=BOX_COLOR)
        draw.text((box["xmin"], box["ymin"]), text, fill=TEXT_COLOR, font=font)

    return image


def run(image_path: str, output_path: str = "output.jpg", threshold: float = THRESHOLD):
    """Full pipeline: load → detect → annotate → save."""
    feature_extractor, model = load_model()
    recognizer = load_ocr()

    print(f"Processing: {image_path}")
    image, detections = detect_plates(image_path, feature_extractor, model, threshold)

    # Run OCR on each detected plate
    for det in detections:
        ocr = read_plate_text(image, det["box"], recognizer)
        det["text"] = ocr["text"]
        det["region"] = ocr["region"]

    # Print results
    if not detections:
        print("No license plates detected above threshold.")
    else:
        print(f"Detected {len(detections)} plate(s):\n")
        for i, det in enumerate(detections, 1):
            b = det["box"]
            print(f"  [{i}] Label  : {det['label']}")
            print(f"       Text   : {det['text']!r}")
            if det.get("region"):
                print(f"       Region : {det['region']}")
            print(f"       Score  : {det['score']:.4f} ({det['score']:.2%})")
            print(f"       Box    : xmin={b['xmin']}, ymin={b['ymin']}, "
                  f"xmax={b['xmax']}, ymax={b['ymax']}\n")

    # Annotate and save
    annotated = draw_detections(image.copy(), detections)
    annotated.save(output_path)
    print(f"Annotated image saved to: {output_path}")

    return detections


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yolos_license_plate.py <image_path> [output_path] [threshold]")
        print("  image_path   : path to input image  (required)")
        print("  output_path  : path to save output  (default: output.jpg)")
        print("  threshold    : confidence cutoff     (default: 0.5)")
        sys.exit(1)

    img_path  = sys.argv[1]
    out_path  = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"
    conf      = float(sys.argv[3]) if len(sys.argv) > 3 else THRESHOLD

    run(img_path, out_path, conf)
