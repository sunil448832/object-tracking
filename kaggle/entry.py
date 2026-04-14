"""Kaggle kernel entry point.

Kaggle script kernels only upload this single file. Real code and data live in
two mounted datasets:
  /kaggle/input/object-tracking-code/   (tracking.zip or pre-extracted files)
  /kaggle/input/object-tracking-data/   (videos)
Outputs must be written under /kaggle/working/ to be retrievable.
"""
import os
import shutil
import subprocess
import sys
import zipfile

CODE_SRC = "/kaggle/input/object-tracking-code"
DATA_IN = "/kaggle/input/object-tracking-data"
WORK = "/kaggle/working"
CODE_ROOT = os.path.join(WORK, "code")
OUT_DIR = os.path.join(WORK, "results")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CODE_ROOT, exist_ok=True)

# Dataset may arrive as tracking.zip (raw) or pre-extracted flat files.
zips = [os.path.join(CODE_SRC, f) for f in os.listdir(CODE_SRC) if f.endswith(".zip")]
if zips:
    for z in zips:
        stem = os.path.splitext(os.path.basename(z))[0]
        dest = os.path.join(CODE_ROOT, stem)
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
else:
    shutil.copytree(CODE_SRC, os.path.join(CODE_ROOT, "tracking"), dirs_exist_ok=True)

print(f"[entry] code tree: {sorted(os.listdir(CODE_ROOT))}")

sys.path.insert(0, CODE_ROOT)

req = os.path.join(CODE_ROOT, "tracking", "requirements.txt")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req])

print("[entry] data tree (top 2 levels):")
for name in sorted(os.listdir(DATA_IN)):
    full = os.path.join(DATA_IN, name)
    is_dir = os.path.isdir(full)
    print(f"   {name}{'/' if is_dir else ''}")
    if is_dir:
        for q in sorted(os.listdir(full))[:20]:
            print(f"      {q}")

VIDEO_NAME = "NVR-ABD6_ch50_main_20260409092100_20260409110059.mp4"
matches = []
for root, _dirs, files in os.walk(DATA_IN):
    if VIDEO_NAME in files:
        matches.append(os.path.join(root, VIDEO_NAME))
if not matches:
    all_mp4 = []
    for root, _dirs, files in os.walk(DATA_IN):
        for f in files:
            if f.endswith(".mp4"):
                all_mp4.append(os.path.join(root, f))
    print(f"[entry] all .mp4 found ({len(all_mp4)}):")
    for m in all_mp4:
        print(f"   {m}")
assert matches, f"{VIDEO_NAME} not found under {DATA_IN}"
video = matches[0]
output = os.path.join(
    OUT_DIR, "NVR-ABD6_ch50_main_20260409092100_20260409110059_output_r50.mp4"
)
print(f"[entry] input : {video}")
print(f"[entry] output: {output}")

subprocess.check_call(
    [
        sys.executable, os.path.join(CODE_ROOT, "tracking", "run_tracking.py"),
        video,
        "--output", output,
        "--frames", "0",
        "--stride", "2",
        "--threshold", "0.7",
        "--ocr-every", "1",
        "--min-vehicle-area", "8000",
        "--min-ocr-conf", "0.7",
        "--lock-in-conf", "0.95",
    ],
    cwd=CODE_ROOT,
)
print("[entry] done. outputs in /kaggle/working/results/")
