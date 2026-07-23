# Real vs Fake Face Detection — Master 2 Thesis

Computer vision project on detecting real (photographed) vs. fake (AI-generated)
face images, as part of a Master 2 thesis. This repository tracks the practical
implementation work, alongside the thesis's literature review.

## Current status: Baseline #1 — complete

A first end-to-end pipeline for **digital face forgery (deepfake) detection**
has been built and evaluated:

- **Model**: ResNet18 pretrained on ImageNet, backbone frozen, final layer fine-tuned
- **Dataset**: [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (Kaggle) — real photographs (FFHQ) vs. StyleGAN-generated faces
- **Result**: 80.8% test accuracy, 0.888 AUC (see `results/` for full details)

See `Baseline_Results_Summary.pdf` for the 1-page overview, or `baseline_colab.ipynb`
for the full code and training/evaluation outputs.

## Repository structure

```
├── README.md                          ← this file
├── baseline_colab.ipynb               ← full Colab notebook: data pipeline,
│                                          training, evaluation (code + outputs)
├── results/
│   ├── Baseline_Results_Summary.pdf   ← 1-page results overview (for quick review)
│   └── Baseline_Results_Summary.docx  ← same, editable Word version
```

## How to reproduce

1. Open `baseline_colab.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU
3. Run all cells in order (requires a free Kaggle account + API token for
   dataset download — instructions are in the notebook)

## Roadmap / next steps

- [ ] Fine-tune deeper layers (unfreeze last residual block) to improve past
      the current ~81% plateau
- [ ] Cross-dataset generalization test (train on this dataset, test on a
      different fake-face source — e.g. a different GAN or diffusion-generated set)
- [ ] Extend to video-based deepfakes (FaceForensics++) and/or explore the
      face anti-spoofing (presentation attack) track, per the thesis's state-of-the-art review
