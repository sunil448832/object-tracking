"""
Re-ID appearance embedder — DINOv2 ViT-S/14 with registers.

Produces L2-normalized CLS-token embeddings (384-D) for a batch of object crops.
Used by the tracker to match detections across frames through occlusion.

License: DINOv2 is Apache 2.0 (Meta). Safe for commercial use.

Standalone test:
    python -m tracking.reid <image_path> [image_path ...]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "dinov2_vits14_reg"   # 21M params, 384-D output
INPUT_SIZE = 224                   # must be a multiple of 14
# Standard ImageNet normalization — what DINOv2 was trained with
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class DinoV2Embedder:
    """Wraps a DINOv2 backbone to produce re-ID embeddings."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        input_size: int = INPUT_SIZE,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.input_size = input_size
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"[ReID] Loading {model_name} on {self.device} ...")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        self.embed_dim = self.model.embed_dim
        print(f"[ReID] Loaded. Embedding dim = {self.embed_dim}")

        # Preprocess: resize short side to input_size, center-crop square, normalise.
        # Aspect-ratio-preserving resize + centre-crop works well for vehicle crops.
        self.preprocess = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ── Public API ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed(self, crops: Sequence[Image.Image | np.ndarray]) -> np.ndarray:
        """
        Compute L2-normalized embeddings for a list of image crops.

        Args:
            crops: list of PIL.Image (RGB) or HxWx3 uint8 numpy arrays (RGB).

        Returns:
            np.ndarray of shape (N, embed_dim), float32, L2-normalised (rows are unit vectors).
        """
        if len(crops) == 0:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        tensors = [self._to_tensor(c) for c in crops]
        batch = torch.stack(tensors).to(self.device)

        features = self.model(batch)                      # (N, D), CLS token
        features = F.normalize(features, p=2, dim=-1)     # L2 normalise
        return features.cpu().numpy().astype(np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────
    def _to_tensor(self, crop: Image.Image | np.ndarray) -> torch.Tensor:
        if isinstance(crop, np.ndarray):
            crop = Image.fromarray(crop)
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        return self.preprocess(crop)


# ── Standalone test ──────────────────────────────────────────────────────────
def _test(image_paths: list[str]) -> None:
    """Load each image, embed it, and print pairwise cosine similarities."""
    if not image_paths:
        print("Usage: python -m tracking.reid <image1> [image2 ...]")
        sys.exit(1)

    imgs = []
    for p in image_paths:
        if not Path(p).exists():
            print(f"[ERROR] Not found: {p}")
            sys.exit(1)
        imgs.append(Image.open(p).convert("RGB"))
        print(f"[TEST] Loaded {p}  size={imgs[-1].size}")

    embedder = DinoV2Embedder()
    feats = embedder.embed(imgs)
    print(f"\n[TEST] Embedding matrix: shape={feats.shape}  dtype={feats.dtype}")
    print(f"[TEST] L2 norms (should all be ~1.0): {np.linalg.norm(feats, axis=1)}")

    if len(imgs) > 1:
        sim = feats @ feats.T  # pairwise cosine since unit-normed
        print("\n[TEST] Pairwise cosine similarity matrix:")
        header = "        " + "  ".join(f"img{i}" for i in range(len(imgs)))
        print(header)
        for i in range(len(imgs)):
            row = "  ".join(f"{sim[i, j]:+.3f}" for j in range(len(imgs)))
            print(f"  img{i}  {row}")
        print("\n  Interpretation: 1.0 = identical image, values near 1 mean visually similar.")


if __name__ == "__main__":
    _test(sys.argv[1:])
