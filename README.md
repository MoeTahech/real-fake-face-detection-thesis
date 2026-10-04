# Real vs Fake Face Detection — Master 2 Thesis

Computer vision project on detecting real (photographed) vs. fake (AI-generated)
face images, as part of a Master 2 thesis. This repository tracks the practical
implementation work, alongside the thesis's literature review.

## Current status: Phases 1–9 complete

| Phase | Description | Key Result |
|---|---|---|
| 1 — Baseline | ResNet18, frozen backbone, linear probe | 80.8% acc, 0.888 AUC |
| 2 — Deeper fine-tuning | Unfreeze `layer4` of ResNet18 | 98.8% acc, 0.999 AUC |
| 3 — Cross-dataset evaluation | Test on independent face-swap deepfake dataset (no retraining) | ~49% acc, ~0.50 AUC (near chance) |
| 4 — Ensembling | ResNet18 + EfficientNet-B0, probability averaging | Does not fix generalization gap |
| 5 — Adversarial robustness | FGSM / PGD attacks on Phase 2 model | Accuracy collapses to 0% at epsilon=0.01 |
| 6 — Mixed-source training | Fine-tune on combined StyleGAN + face-swap data | Face-swap AUC recovers from 0.525 -> 0.928 |
| 7 — Third-domain test (zero-shot) | Phase 2 & 6 models tested on unseen diffusion-generated faces | AUC: Phase 2 = 0.277 (below chance), Phase 6 = 0.552 (partial fix) |
| 8 — Frequency-domain detection | ResNet18 trained on FFT spectra instead of RGB pixels | Small generalization gain (+0.03 AUC), lower in-distribution accuracy |
| 9 — Three-source training (few-shot) | Train with StyleGAN + face-swap + 1,500 diffusion images | Diffusion AUC jumps from 0.553 (Phase 6) to **0.999** |

**Datasets used:**
- [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (Kaggle) — real photographs (FFHQ) vs. StyleGAN-generated faces.
- [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) (Kaggle) — real face-swap deepfake video frames.
- [Face Dataset Using Stable Diffusion v.1.4](https://www.kaggle.com/datasets/bwandowando/faces-dataset-using-stable-diffusion-v14) (Kaggle) — Stable Diffusion-generated faces.

**Overall finding:** a fine-tuned CNN achieves near-perfect in-distribution
accuracy (98.8%) on the generator family it was trained on, but generalizes
very poorly, zero-shot, to unseen forgery types (Phase 3, Phase 7) — a
problem that neither architectural ensembling (Phase 4) nor a change of input
representation to the frequency domain (Phase 8) meaningfully fix. Training
data diversity (Phase 6) is a substantially more effective remedy, though its
zero-shot benefit only partially extends to a third, completely unrelated
forgery paradigm. The most important finding of this work (Phase 9) is that
this remaining gap is **not a fundamental model limitation**: including just
1,500 training images of the third (diffusion) domain — a small fraction of
the training set — resolved detection on that domain almost completely
(AUC 0.553 -> 0.999), at negligible cost to performance on the other two
domains. This indicates the generalization limitations observed throughout
this work stem primarily from an absence of relevant training examples,
not from any fundamental architectural or representational constraint.
Adversarial robustness (Phase 5) remains an entirely separate, unaddressed
weakness throughout. These results provide direct empirical evidence for,
and a granular characterization of, the central open challenge identified in
the thesis's state-of-the-art review (Section 8).

See `results/Final_Results_Report.pdf` for the full write-up with confusion
matrices, robustness curves, and per-phase interpretation, or
`results/Methodology_and_Results.docx` for the formal thesis-chapter write-up
(Methodology and Results chapters covering Phases 1-7). The individual
notebooks in `notebooks/` contain the complete code and training/evaluation
outputs.

## Repository structure

```
├── README.md                          ← this file
├── notebooks/
│   ├── 01_baseline.ipynb              ← ResNet18 frozen-backbone baseline
│   ├── 02_finetuning.ipynb            ← ResNet18 deeper fine-tuning (layer4)
│   ├── 03_cross_dataset_eval.ipynb    ← generalization / drift evaluation
│   ├── 04_ensemble.ipynb              ← ResNet18 + EfficientNet-B0 ensemble
│   ├── 05_adversarial_robustness.ipynb ← FGSM / PGD robustness evaluation
│   ├── 06_mixed_source_training.ipynb ← combined StyleGAN + face-swap training
│   ├── 07_third_domain_test.ipynb     ← zero-shot generalization test (diffusion)
│   ├── 08_frequency_domain.ipynb      ← FFT-based input representation experiment
│   └── 09_three_source_training.ipynb ← few-shot recovery via 3-source training
└── results/
    ├── Final_Results_Report.pdf       ← full write-up, all 9 phases (for quick review)
    ├── Final_Results_Report.docx      ← same, editable Word version
    ├── Methodology_and_Results.docx   ← formal thesis-chapter draft (Phases 1-7)
    └── Methodology_and_Results.pdf    ← same, PDF version
```

## How to reproduce

Each notebook is self-contained and can be opened directly in Google Colab.
Requires a free Kaggle account + API token.

Run order: `01_baseline` → `02_finetuning` → `03_cross_dataset_eval` →
`04_ensemble` → `05_adversarial_robustness` → `06_mixed_source_training` →
`07_third_domain_test` → `08_frequency_domain` → `09_three_source_training`.
Each notebook saves its trained model checkpoint to Google Drive
(`My Drive/thesis_checkpoints/`), so later notebooks can load earlier
checkpoints without retraining. Note: Phase 9 carefully splits the diffusion
dataset into disjoint training/evaluation portions to avoid data leakage with
Phases 7-8's evaluation.

## Roadmap / next steps

- [ ] Adversarial training (or other defense) to address the Phase 5
      robustness gap, unaddressed by any phase to date
- [ ] Characterize the few-shot learning curve suggested by Phase 9: repeat
      with 100, 500, and 1,000 diffusion training images (instead of 1,500)
      to find the minimum sample size needed for substantial recovery
- [ ] CLIP-based or other foundation-model detector, as a comparison point
      against the data-centric remedy established as most effective in this work
- [ ] Feature-level investigation of the Phase 7/8 below-chance AUC finding
      as a focused, self-contained sub-question
