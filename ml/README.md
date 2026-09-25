# ML (training happens in Google Colab / Kaggle, not on our PCs)

- `count_dataset.py` — dataset audit, cleaning and leakage-free 70/15/15 re-split (manuscript Sections 6–7)
- `train_evaluate.py` — training, evaluation and TFLite export for all six models (Sections 9–11)
- The dataset (8,898 images) and result folders stay in Google Drive (`MyDrive/IsdaYou/`), not in Git.

Deployed model: **EfficientNet-Lite0, FP16**, 6.5 MB → `mobile-app/app/src/main/assets/species_classifier.tflite`.
Input: 1 × 224 × 224 × 3, raw RGB 0–255 (normalization is inside the model). Output: 31 softmax probabilities in `labels.txt` order.
Primary threshold 0.45; alternatives ≥ 0.10 (max 3). Backup model: DenseNet121 FP16 (13.4 MB), same input and output.
