# Experiment 2 results

## Controlled classifier comparison

Fixed Experiment 1 split (50,000/10,000); official test set used only after validation checkpoint selection. Same training budget in every run: 30 epochs, batch 128, Adam lr=0.001; three training seeds: 42, 123, 2026. Pixel permutation seed 20261024 is applied identically to train, validation, and test images.

| Input | Model | Parameters | Test accuracy (mean ± sample SD) | Train seconds (mean ± sample SD) |
|---|---|---:|---:|---:|
| original | cnn | 421,642 | 0.9189 ± 0.0034 | 132.3 ± 0.5 |
| original | mlp | 101,770 | 0.8840 ± 0.0012 | 130.4 ± 1.0 |
| permuted | cnn | 421,642 | 0.8788 ± 0.0023 | 152.4 ± 0.2 |
| permuted | mlp | 101,770 | 0.8859 ± 0.0010 | 151.9 ± 1.5 |

Three-seed statistics are in `results/exp2/comparison_statistics.csv`; every single-run metric is in `comparison_all_seeds.csv`. Layer sizes and parameter counts are in `architecture.csv`.

## Diffusion model

Unconditional DDPM trained on 50,000 fixed training examples, 32 channels, 100 steps, 30 epochs, device cuda; training took 129.58 s. Fixed seeds: train 42, sampling 20261024, trajectory noise 314159. Synchronized complete sampling took 0.205 s for 100 samples (0.00205 s/image). Noise-prediction MSE: 0.4021 → 0.1402. Exact initial Gaussian noise is saved in `results/exp2/diffusion/fixed_samples.pt`.

Evaluation caveat: an independently trained CNN proxy predicts T-shirt/top 3, Trouser 3, Pullover 1, Sandal 2, Shirt 5, Bag 86 for the fixed 100-sample batch. Most samples are classified as Bag and multiple classes are absent, showing severe mode collapse/poor category coverage. These are proxy labels rather than human ground truth; the complete sample grid, nearest neighbors, and fixed-batch bottom-quartile confidence failures are included. Nearest neighbors use 10,000 examples from the fixed train subset (every fifth image).

## Files and reproducibility

Runnable source/commands: `exp2/` and `exp2/README.md`. Histories, metrics, per-class recall, confusion matrices, fixed-order errors, and logs: `results/exp2/`. Checkpoints: `checkpoints/exp2/`. Figures: `report/figures/exp2/`. Six-page PDF: `report/experiment2_report.pdf`. Download Fashion-MNIST with `python3 exp2/prepare_data.py --download`; the exact split file is included in the project bundle. A 10,000-sample/5-epoch diffusion pilot is retained separately; the reported DDPM is the 50,000-sample/30-epoch run.
