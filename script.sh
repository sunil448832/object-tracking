python -m tracking.run_tracking \
  "./data/images/NVR-ABD6_ch50_main_20260409092100_20260409110059.mp4" \
  --output "./data/results/NVR-ABD6_ch50_main_20260409092100_20260409110059_output_r50.mp4" \
  --frames 0 \
  --stride 2 \
  --threshold 0.7 \
  --ocr-every 1 \
  --min-vehicle-area 8000 \
  --min-ocr-conf 0.7 \
  --lock-in-conf 0.95


bash kaggle/push_data.sh      # uploads data/ as private dataset (once, then re-run on data changes)
bash kaggle/push_kernel.sh    # stages code + pushes kernel → runs on Kaggle GPU
bash kaggle/pull_output.sh    # downloads /kaggle/working/ → ./kaggle_output/
bash colab/pack_bundle.sh
kaggle kernels status sharma37/object-tracking-run