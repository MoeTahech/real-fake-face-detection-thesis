# Deepfake Detection Using Computer Vision: Generalization and Reliability Across Generators

Code and results of a Master 2 thesis on **single-image detection of AI-generated faces**.
The thesis asks a simple question: a detector that is almost perfect on the generator it
was trained on, how does it behave on generators it has never seen, and how much can the
reported numbers be trusted?

Three forgery families are studied, each with one public dataset:

- **StyleGAN** faces (fully generated), used for training and for the in-distribution test;
- **face swaps** (a GAN-synthesized face blended into a real photograph);
- **Stable Diffusion** faces (DiffusionFace, SD v1.5 and v2.1).

The main model is a ResNet18 fine-tuned on StyleGAN faces. It is compared with EfficientNet-B0,
a frequency-domain (FFT) model, and frozen CLIP backbones (ViT-B/32 and ViT-L/14).

> Thesis PDF: `thesis/Thesis.pdf` (add the final version here).

## Research questions

| RQ | Question | Phases |
|---|---|---|
| RQ1 | How well does a detector work in distribution, and what role does fine-tuning play? | 1, 2, ablation |
| RQ2 | What happens on generators not seen in training (face swaps, diffusion)? | 3, 7b |
| RQ3 | Can the gap be closed (ensemble, FFT input, CLIP, mixed training)? | 4, 6, 8, 9, 9b |
| RQ4 | How reliable are the conclusions (seeds, shortcuts, degradations, attacks)? | 5, 10, 11, 12 and analyses A, D |

## Main results

AUC of each model on the same frozen test subsets (5,000 + 5,000 images for StyleGAN and
face swaps; 3,000 + 3,000 for DiffusionFace, with its own real images). 0.50 is chance.

| Model | Basis | StyleGAN | Face swap | DiffusionFace SD v1.5 | DiffusionFace SD v2.1 |
|---|---|---|---|---|---|
| Phase 1: frozen ResNet18 | single run | 0.890 | 0.490 | 0.535 | 0.496 |
| Phase 2: ResNet18, layer4 fine-tuned | 3-seed mean | 0.999 | 0.502 | 0.528 | 0.530 |
| Phase 4: EfficientNet-B0 | single run | 0.986 | 0.500 | 0.496 | 0.492 |
| Phase 4: ensemble | single run | 0.999 | 0.499 | 0.498 | 0.492 |
| Phase 6: StyleGAN + face swap | 3-seed mean | 0.996 | 0.898 | 0.496 | 0.493 |
| Phase 8: frequency (FFT) input | single run | 0.917 | 0.559 | 0.319 | 0.290 |
| Phase 9: CLIP ViT-B/32 + linear head | single run | 0.987 | 0.486 | 0.303 | 0.387 |
| CLIP ViT-L/14 + linear head | single run | 0.999 | 0.534 | 0.576 | 0.570 |

**What the results show (in these experiments, on these datasets):**

- Near-perfect accuracy on the training generator (98.6% on average over three seeds) does
  not carry over: the same model is at chance on face swaps and on diffusion faces.
- An ensemble, a frequency-domain input and a frozen CLIP backbone do not close the gap.
  Training on face swaps raises face-swap AUC from 0.50 to 0.90, but only for face swaps,
  and costs about 1.6 points of accuracy on StyleGAN.
- The larger CLIP (ViT-L/14) is the only model slightly above chance on diffusion faces
  (AUC about 0.57). The signal is weak and was measured with a single trained head.
- Some models fall **below** chance on diffusion faces (AUC 0.29 to 0.39): they rank
  diffusion fakes as more real than the real images.
- The main effects hold in every seed. A near-duplicate check of the face-swap split
  (0.9% of test images flagged) leaves the Phase 6 result unchanged.
- The undefended model tolerates moderate JPEG compression and blur but not strong blur or
  heavy downscaling, and a small white-box adversarial perturbation (epsilon = 0.01) drives
  its accuracy to 0%, also after 8-bit rounding and JPEG compression.

Conclusion of the thesis: accuracy on the training generator is not evidence of
generalization, and detectors should be evaluated on generators they have not seen.

## Experiments

Phases are presented by research question in the thesis; their numbers follow the run order.

| Phase | Description | Key result |
|---|---|---|
| 1 | ResNet18, frozen backbone, linear probe | 80.8% accuracy, AUC 0.888 on StyleGAN |
| 2 | ResNet18 with `layer4` unfrozen (plus ablation of fine-tuning vs learning rate) | 98.6% accuracy (3-seed mean), AUC 0.999 |
| 3 | Phase 2 model tested on face swaps, no retraining | about 49 to 50% accuracy, AUC about 0.50 |
| 4 | EfficientNet-B0 and probability-averaging ensemble | no improvement on unseen domains |
| 5 | FGSM and PGD attacks on the Phase 2 model | 0% accuracy at epsilon = 0.01 |
| 6 | Training on StyleGAN + face swaps | face-swap AUC 0.898 (3-seed mean), StyleGAN accuracy 96.9% |
| 7b | RGB models tested on DiffusionFace | AUC 0.49 to 0.53 (chance) |
| 8 | ResNet18 on FFT spectra | 83.8% on StyleGAN, below chance on DiffusionFace |
| 9 | Frozen CLIP ViT-B/32 + linear head | 94.4% on StyleGAN, below chance on DiffusionFace |
| 10 | Seeds 0, 1, 2 and bootstrap confidence intervals | main effects hold in every seed |
| 11 | Shortcut and confound audit (file facts, pixel statistics) | a residual file-size difference remains in DiffusionFace |
| 12 | Everyday degradations (JPEG, blur, resize, noise, combined) | tolerates moderate degradation, fails on strong blur and downscaling |

### Additional analyses (notebooks A to D)

| Notebook | What it does | Result |
|---|---|---|
| A, leakage check | Near-duplicate search between the face-swap train and test folders (CLIP similarity) | 97 of 10,905 test images (0.9%) flagged; Phase 6 AUC 0.928 with and without them |
| B, frozen test suite | Scores every model on exactly the same test images | the table above |
| C, larger CLIP | CLIP ViT-L/14 and a ViT-B/32 re-run, final embedding and intermediate layers | AUC 0.576 / 0.570 on DiffusionFace; chance on face swaps (0.534) |
| D, adversarial re-run and controls | Attack after 8-bit rounding and JPEG; StyleGAN-only control runs (30,000 and 60,000 images) | the attack survives; the StyleGAN cost of Phase 6 comes from having fewer StyleGAN images |

## Datasets

| Dataset | Forgery type | Used for |
|---|---|---|
| [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (Kaggle) | StyleGAN vs real FFHQ | training, validation, in-distribution test |
| [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) (Kaggle), derived from OpenForensics | face swap (GAN face blended into a real photo) | cross-domain test; training source in Phase 6 |
| [DiffusionFace](https://github.com/Rapisurazurite/DiffFace) (Chen et al., 2024; [arXiv:2403.18471](https://arxiv.org/abs/2403.18471)) | Stable Diffusion v1.5 and v2.1 faces, with real CelebA-HQ images | test only |

An earlier Stable Diffusion v1.4 dataset from Kaggle was excluded: it contains only fake
images, so it has no real images of its own and cannot give a fair real-versus-fake test.

## Repository structure

```
├── README.md
├── thesis/
│   └── Thesis.pdf                          <- final thesis
└── notebooks/
    ├── 01_baseline.ipynb                   Phase 1   frozen ResNet18
    ├── 02_finetuning.ipynb                 Phase 2   layer4 unfrozen
    ├── 02b_ablation.ipynb                  ablation  fine-tuning vs learning rate
    ├── 03_cross_dataset_eval.ipynb         Phase 3   face-swap evaluation
    ├── 04_ensemble.ipynb                   Phase 4   EfficientNet-B0 and ensemble
    ├── 05_adversarial_robustness.ipynb     Phase 5   FGSM / PGD
    ├── 06_mixed_source_training.ipynb      Phase 6   StyleGAN + face swap
    ├── 07b_diffusionface_external_check.ipynb   Phase 7b  DiffusionFace, RGB models
    ├── 08_frequency_domain.ipynb           Phase 8   FFT-input ResNet18
    ├── 08b_frequency_diffusionface_check.ipynb  Phase 8   DiffusionFace, frequency model
    ├── 10_clip_based_detector.ipynb        Phase 9   frozen CLIP ViT-B/32
    ├── 11_seeds_and_confidence_intervals.ipynb  Phase 10  seeds and bootstrap
    ├── 13_shortcut_confound_check.ipynb    Phase 11  shortcut audit
    ├── 14_degradation_robustness.ipynb     Phase 12  everyday degradations
    ├── A_leakage_check.ipynb               near-duplicate check
    ├── B_frozen_test_suite.ipynb           all models on the same test subsets
    ├── C_clip_vitl14.ipynb                 larger CLIP follow-up
    └── D_adversarial_and_controls.ipynb    attack after 8-bit/JPEG, control runs
```

Notebook numbers do not match phase numbers (the notebook names above give the
correspondence).

## How to reproduce

1. Open a notebook directly in Google Colab (a single T4 GPU is enough).
2. Create a free Kaggle account and an API token. **Never commit the token**: upload
   `kaggle.json` to the Colab session or use Colab secrets, and keep it out of the notebooks.
3. Mount Google Drive. Trained checkpoints are saved to `My Drive/thesis_checkpoints/`,
   so later notebooks load earlier checkpoints without retraining. Notebook D also caches
   its datasets on Drive, so they are downloaded only once.
4. Run order: `01` -> `02` (and `02b`) -> `03` -> `04` -> `05` -> `06` -> `07b` -> `08` and
   `08b` -> `10` -> `11` -> `13` -> `14`, then `A` -> `B` -> `C` -> `D`.

Requirements: Python, PyTorch and torchvision, Hugging Face `transformers` (CLIP
`openai/clip-vit-base-patch32` and `openai/clip-vit-large-patch14`), scikit-learn, Pillow.
Library versions of the Colab environment were not pinned, and the Kaggle datasets are
maintained by third parties, so exact numbers may differ slightly when rerun, especially
for the single, unseeded runs.

## Limitations

- Each forgery type is represented by one public dataset; the face-swap data has
  undocumented processing and an undocumented train/test split.
- The diffusion evidence comes from Stable Diffusion v1.5 and v2.1, which are closely related.
- Phases 4, 5, 8 and 9 are single runs; the CLIP heads were trained once, and re-training
  the small head changed its DiffusionFace AUC by 0.09 to 0.10.
- The alternatives to ResNet18 were not tuned, and training used almost no augmentation.
- No standard benchmark (FaceForensics++, Celeb-DF, DF40, GenImage) was used, so the
  numbers cannot be compared directly with published results.
- Only an undefended model was attacked, in a white-box setting.

## Future work

- Leave-one-generator-out training with more generators.
- A stronger CLIP test: several seeds for the linear head, lightweight adapters, training on
  more than one generator.
- Training with blur and JPEG augmentation, then repeating the cross-domain and robustness tests.
- Evaluation on standard benchmarks, and a face-swap split grouped by source photograph.
- Adversarial training and attacks that account for JPEG compression.

## Citation

```
[Author name]. Deepfake Detection Using Computer Vision Techniques:
Generalization and Reliability Across Generators. Master 2 thesis, [University], [year].
```

## License

[Add a license for the code. The datasets keep their own licenses; DiffusionFace is
released under CC BY 4.0 for research use.]
