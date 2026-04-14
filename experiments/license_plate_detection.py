"""
License Plate Detection using nickmuchi/detr-resnet50-license-plate-detection
Model: https://huggingface.co/nickmuchi/detr-resnet50-license-plate-detection
License: Apache 2.0 (Commercial-friendly)

Install dependencies:
    pip install transformers torch pillow matplotlib
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME  = "nickmuchi/detr-resnet50-license-plate-detection"
IMAGE_PATH  = "car.jpg"          # <-- Change to your image file path
THRESHOLD   = 0.5                # Confidence threshold (0.0 – 1.0)
OUTPUT_PATH = "output.jpg"       # Annotated image save path


# ── Load Model & Processor ───────────────────────────────────────────────────

print(f"Loading model: {MODEL_NAME} ...")
processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
model     = DetrForObjectDetection.from_pretrained(MODEL_NAME)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Running on: {device}")


# ── Load & Preprocess Image ──────────────────────────────────────────────────

image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image loaded: {IMAGE_PATH}  |  Size: {image.size[0]}x{image.size[1]} px")

# Prepare inputs for the model
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}


# ── Run Inference ────────────────────────────────────────────────────────────

print("Running inference ...")
with torch.no_grad():
    outputs = model(**inputs)


# ── Post-process Results ─────────────────────────────────────────────────────

# Convert raw outputs → bounding boxes in (x1, y1, x2, y2) pixel format
target_sizes = torch.tensor([image.size[::-1]])   # (height, width)
results = processor.post_process_object_detection(
    outputs,
    target_sizes=target_sizes,
    threshold=THRESHOLD
)[0]

scores = results["scores"].cpu()
labels = results["labels"].cpu()
boxes  = results["boxes"].cpu()

print(f"\nDetections above threshold ({THRESHOLD}):")

if len(scores) == 0:
    print("  No license plates detected. Try lowering THRESHOLD.")
else:
    for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
        x1, y1, x2, y2 = [round(v.item(), 1) for v in box]
        label_name = model.config.id2label[label.item()]
        print(f"  [{i+1}] {label_name}  |  confidence: {score:.2%}  |  box: [{x1}, {y1}, {x2}, {y2}]")


# ── Visualise & Save ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

colors = plt.cm.get_cmap("tab10").colors

for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
    x1, y1, x2, y2 = box.tolist()
    width  = x2 - x1
    height = y2 - y1
    color  = colors[i % len(colors)]

    # Draw bounding box
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=2, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)

    # Draw label + confidence
    label_name = model.config.id2label[label.item()]
    ax.text(
        x1, y1 - 6,
        f"{label_name}: {score:.2%}",
        color="white", fontsize=10, fontweight="bold",
        bbox=dict(facecolor=color, alpha=0.75, pad=2, edgecolor="none")
    )

ax.set_title("License Plate Detection — nickmuchi/detr-resnet50-license-plate-detection", fontsize=12)
ax.axis("off")
plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nAnnotated image saved → {OUTPUT_PATH}")
plt.show()
