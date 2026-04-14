python run_tracking.py \
  "../data/images/NVR-ABD6_ch50_main_20260409092100_20260409092138.mp4" \
  --output ../data/results/NVR-ABD6_ch50_main_20260409092100_20260409092138_output_r50.mp4 \
  --frames 0 \
  --stride 2 \
  --threshold 0.7 \
  --ocr-every 1 \
  --min-vehicle-area 8000 \
  --min-ocr-conf 0.7 \
  --lock-in-conf 0.95