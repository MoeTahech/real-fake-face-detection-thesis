# Real vs Fake Face Detection — Master 2 Thesis

Computer vision project on detecting real (photographed) vs. fake (AI-generated)
face images, as part of a Master 2 thesis. This repository tracks the practical
implementation work, alongside the thesis's literature review.

## Current status: Phases 1–6 complete

| Phase | Description | Key Result |
|---|---|---|
| 1 — Baseline | ResNet18, frozen backbone, linear probe | 80.8% acc, 0.888 AUC |
| 2 — Deeper fine-tuning | Unfreeze `layer4` of ResNet18 | 98.8% acc, 0.999 AUC |
| 3 — Cross-dataset evaluation | Test on independent face-swap deepfake dataset (no retraining) | ~49% acc, ~0.50 AUC (near chance) |
| 4 — Ensembling | ResNet18 + EfficientNet-B0, probability averaging | Does not fix generalization gap |
| 5 — Adversarial robustness | FGSM / PGD attacks on Phase 2 model | Accuracy collapses to 0% at epsilon=0.01 |
| 6 — Mixed-source training | Fine-tune on combined StyleGAN + face-swap data | Face-swap AUC recovers from 0.525 -> 0.928 |

**Dataset (Phases 1, 2, 4, 5, 6):** [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (Kaggle) — real photographs (FFHQ) vs. StyleGAN-generated faces.

**Dataset (Phases 3, 6 cross-domain):** [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) (Kaggle) — real face-swap deepfake video frames.

**Overall finding:** a fine-tuned CNN achieves near-perfect in-distribution
accuracy (98.8%) on the generator family it was trained on, but this does not
reflect a robust, deployable deepfake detector on its own. The same model
collapses to near-chance accuracy on an unseen forgery type (Phase 3), and
ensembling with a second architecture does not recover this generalization
loss since both models share the same single-source blind spot (Phase 4). It
is also almost completely defeated by imperceptible adversarial perturbations
(Phase 5). However, diversifying the training data to include both forgery
types (Phase 6) recovers the large majority of the lost generalization
performance (Face-swap AUC: 0.525 -> 0.928) while retaining strong performance
on the original domain — showing the generalization gap is substantially more
fixable through training data diversity than through architecture alone.
Adversarial robustness remains an open weakness not addressed by mixed-source
training. See the thesis's state-of-the-art review, Section 8, for the
broader literature context.

See `results/Final_Results_Report.pdf` for the full write-up with confusion
matrices, robustness curves, and per-phase interpretation, or the individual
notebooks in `notebooks/` for the complete code and training/evaluation outputs.

## Repository structure

```
├── README.md                          ← this file
├── notebooks/
│   ├── 01_baseline.ipynb              ← ResNet18 frozen-backbone baseline
│   ├── 02_finetuning.ipynb            ← ResNet18 deeper fine-tuning (layer4)
│   ├── 03_cross_dataset_eval.ipynb    ← generalization / drift evaluation
│   ├── 04_ensemble.ipynb              ← ResNet18 + EfficientNet-B0 ensemble
│   ├── 05_adversarial_robustness.ipynb ← FGSM / PGD robustness evaluation
│   └── 06_mixed_source_training.ipynb ← combined StyleGAN + face-swap training
└── results/
    ├── Final_Results_Report.pdf       ← full write-up, all 6 phases (for quick review)
    └── Final_Results_Report.docx      ← same, editable Word version
```

## How to reproduce

Each notebook is self-contained and can be opened directly in Google Colab.
Requires a free Kaggle account + API token, configured via Colab Secrets
(🔑 icon, left sidebar — `KAGGLE_USERNAME` and `KAGGLE_KEY`).

Run order: `01_baseline` → `02_finetuning` → `03_cross_dataset_eval` →
`04_ensemble` → `05_adversarial_robustness` → `06_mixed_source_training`. Each
notebook saves its trained model checkpoint to Google Drive
(`My Drive/thesis_checkpoints/`), so later notebooks can load earlier
checkpoints without retraining.

## Roadmap / next steps

- [ ] Adversarial training (or other defense) to address the Phase 5
      robustness gap, which mixed-source training (Phase 6) did not target
- [ ] Extend mixed-source training to additional forgery families (e.g.
      diffusion-generated faces) to test how far the Phase 6 result generalizes
- [ ] Possible extension: frequency-domain features, a foundation-model-based
      approach (e.g. CLIP adapter), or the face anti-spoofing (presentation
      attack) track, per the thesis's state-of-the-art review
