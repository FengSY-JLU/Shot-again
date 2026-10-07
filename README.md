# Shot-Again

## Self-Supervised Underwater Image Enhancement With Transmission-Space Regularization and Structure-Channel Feature Regulation

Official PyTorch implementation of:

**Shiyao Feng, Chunru Wan, Jun-Hong Cui, and Gaochao Xu,  
"Shot-Again: Self-Supervised Underwater Image Enhancement With Transmission-Space Regularization and Structure-Channel Feature Regulation,"  
IEEE Access, vol. 14, pp. 145820–145838, 2026.**

[![Paper](https://img.shields.io/badge/IEEE%20Access-Paper-blue)](https://ieeexplore.ieee.org/document/11692865)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2026.3734227-blue)](https://doi.org/10.1109/ACCESS.2026.3734227)

## Overview

Shot-Again is a self-supervised underwater image enhancement framework built on re-degradation-based learning. It adds an explicit transmission-space regularization signal and two lightweight feature-regulation modules:

- **Transmission-space regularization** constrains the relative behavior of transmission estimates for the original and proxy re-degraded observations.
- **Structure-Preserving Block (SPB)** provides structure-aware feature regulation through asymmetric convolutions and gated processing.
- **Spatial Group Cross-Channel Attention (SGCA)** performs grouped local cross-channel interaction to regulate degradation-dominated feature responses.

The final model does not require paired clean targets for its self-supervised optimization.

## Paper

- **Journal:** IEEE Access
- **Volume:** 14
- **Pages:** 145820–145838
- **Date of publication:** 16 September 2026
- **DOI:** https://doi.org/10.1109/ACCESS.2026.3734227
- **IEEE Xplore:** https://ieeexplore.ieee.org/document/11692865

## Repository Structure

```text
Shot-again/
├── measurement/          # underwater image quality metrics
├── net/                  # network architecture and losses
├── data.py               # dataset-loader helpers
├── dataset.py            # training/evaluation datasets
├── evaluation.py         # evaluation entry point
├── train.py              # training entry point
├── utils.py              # image/model utilities
├── requirements.txt
└── README.md
```

## Environment

The released code was developed with:

- Python 3.13.2
- PyTorch 2.7.1
- NVIDIA A40 GPU

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Clone the repository:

```bash
git clone https://github.com/FengSY-JLU/Shot-again.git
cd Shot-again
```

> `train.py` currently sets `CUDA_VISIBLE_DEVICES='0'` near the beginning of the file. Change this line if you need to train on another GPU.

## Datasets

The dataset package used by this repository is available here:

**[Download datasets from Google Drive](https://drive.google.com/file/d/1wt-nO6-HIT72p70SIAQtUzyOYYywEImM/view?usp=drive_link)**

The paper evaluates Shot-Again on:

- UIEBD
- RUIE
- TestC60
- UFO120
- EUVP-IM
- OceanDark

Please follow the licenses and citation requirements of the original dataset providers.

## Pretrained Models

Pretrained checkpoints are available here:

**[Download pretrained models from Google Drive](https://drive.google.com/file/d/1etv9S_qYt7uDfg2HhGcjJ-ddAwYKvY8N/view?usp=drive_link)**

The released checkpoints are intended for reproducing the experiments reported in the paper. We recommend keeping the Google Drive files publicly readable (**Anyone with the link – Viewer**) so that no access request is required.

## Training

The published full Shot-Again model uses **SPB**, **SGCA**, and **transmission-space regularization**.

Example:

```bash
python train.py \
  --data_train /path/to/train/images \
  --label_train /path/to/train/images \
  --data_test /path/to/test/images \
  --label_test /path/to/test/labels \
  --Indicator UIEBD \
  --IndicatorPath UIEBD/ \
  --batchSize 1 \
  --lr 1e-4 \
  --patch_size 128 \
  --use_spb \
  --use_sgca \
  --use_Lphys
```

For self-supervised training, `--label_train` can point to the same directory as `--data_train`. The current dataset loader expects both paths, but paired clean targets are not used by the final Shot-Again objective unless the optional legacy `--use_Lref` switch is explicitly enabled.

The transmission-space regularization defaults in `train.py` are:

```text
alpha = 1.0
beta  = 0.2
delta = 0.1
tau   = 0.1
eps   = 1e-6
p     = 2
```

> `--use_Lref` is retained only as a compatibility/experimental switch and is not part of the final formulation reported in the paper.

## Evaluation

### Paired evaluation

For a paired benchmark such as UIEBD:

```bash
python evaluation.py \
  --data_test /path/to/UIEBD/test/image \
  --label_test /path/to/UIEBD/test/label \
  --model /path/to/checkpoint.pth \
  --Indicator UIEBD
```

This reports PSNR, SSIM, UCIQE, and UIQM.

### Unpaired evaluation

For an unpaired benchmark such as RUIE:

```bash
python evaluation.py \
  --data_test /path/to/RUIE/test \
  --model /path/to/checkpoint.pth \
  --Indicator RUIE \
  --no-reference
```

In no-reference mode, PSNR and SSIM are skipped and only UCIQE/UIQM are reported by this script.

The full released architecture uses SPB and SGCA by default. For architectural ablations, `evaluation.py` also supports `--no-use_spb` and `--no-use_sgca`.

Evaluation outputs are written under:

```text
Results_<Indicator>/
└── epoch_<tag>/
    ├── J/          # enhanced/reconstructed radiance
    ├── A/          # background-light approximation
    ├── T/          # transmission estimate
    ├── test/
    ├── label/      # paired evaluation only
    └── metrics/
        └── metrics.txt
```

## Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{Feng2026ShotAgain,
  author={Feng, Shiyao and Wan, Chunru and Cui, Jun-Hong and Xu, Gaochao},
  journal={IEEE Access},
  title={Shot-Again: Self-Supervised Underwater Image Enhancement With Transmission-Space Regularization and Structure-Channel Feature Regulation},
  year={2026},
  volume={14},
  pages={145820--145838},
  doi={10.1109/ACCESS.2026.3734227}
}
```

## Contact

For code, pretrained-model, and reproducibility questions, please open an issue or contact:

**Shiyao Feng** — First author and repository maintainer  
GitHub: [@FengSY-JLU](https://github.com/FengSY-JLU)  
Email: `fengsy22@mails.jlu.edu.cn`

Paper corresponding author: **Jun-Hong Cui** (`junhong_cui@jlu.edu.cn`)
