# LAMRetNet: Lesion-Guided Multi-Label Retinal Disease Classification
 
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-blue)](INSERT_KAGGLE_URL_HERE)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
 
> **Paper:** *LAMRetNet: Lesion-Guided Multi-Label Retinal Disease Classification via
> Hierarchical Vision Transformer with Spatial Cross-Attention Fusion*
>
> Farhan Shahriar, Raufu Umme Rhidi, Fatema Taj Mim
> International University of Business Agriculture and Technology (IUBAT), Dhaka, Bangladesh
 
---
 
## Overview
 
LAMRetNet is a two-stage lesion-guided pipeline for simultaneous classification of
**45 retinal diseases** from fundus images. It combines a dedicated segmentation
network with a hierarchical vision transformer, designed for clinically accountable
retinal screening.
 
**Stage 1 — Lesion Attention Module (LAM)**
- EfficientNet-B0 encoder + U-Net decoder
- Trained on IDRiD pixel-level lesion annotations (54 train / 27 test)
- Produces binary spatial attention maps for 5 lesion types
- Best checkpoint: Val Dice = 0.7307 (epoch 66)
 
**Stage 2 — LAMRetNet Classification**
- Hiera-Small backbone (~35M parameters, 224×224 input)
- LAM attention gate: residual gating modulates transformer features
- SCAF: 4-head scaled dot-product cross-attention fusion (d_k = 192)
- Adaptive Disease-Aware Loss: ASL + √-dampened inverse-frequency weights
- Output: Sigmoid multi-label prediction over 45 disease classes
 
---
 
## Key Results
 
| Dataset | Macro-AUC | Micro-F1 | Notes |
|---------|-----------|----------|-------|
| RFMiD v1 (test) | 0.8202 | 0.5571 | 45-class evaluation |
| RFMiD v1 (5-fold CV mean) | 0.7843 ± 0.0230 | — | StratifiedKFold |
| RFMiD v1 (OOF) | 0.6977 | — | Out-of-fold estimate |
| RFMiD 2.0 | 0.7421 | — | No domain adaptation |
| MuReD (14 shared labels) | 0.9248 | — | Cross-protocol generalization |
 
**Ablation (Macro-AUC on RFMiD v1 test):**
 
| Variant | AUC | Δ vs V1 |
|---------|-----|---------|
| V1: Backbone only | 0.6875 | — |
| V2: Backbone + LAM gate only | 0.6764 | −0.0111 |
| V3: Full LAMRetNet (V1 + LAM + SCAF) | 0.7531 | +0.0656 |
 
**Explainability:**
- Grad-CAM++ mean IoU vs IDRiD GT: 0.1412 ± 0.1308
- MC Dropout (N=20) mean σ: 0.0326
 
---
 
## Architecture
 
```
Fundus Image (224×224×3)
        │
        ├──► LAM (EfficientNet-B0 + U-Net) ──► Attention Map â
        │                                            │
        └──► Hiera-Small Backbone ──► Features x_f  │
                                            │        │
                                    Residual Gate: x̂_f = x_f ⊙ (1 + â)
                                            │
                                    SCAF (4-head cross-attention)
                                            │
                              GAP → Dropout(0.3) → Linear(768→256) → ReLU
                                            │
                                    Linear(256→45) → Sigmoid
                                            │
                               Multi-label Disease Predictions
```
 
---
 
## Dataset
 
| Dataset | Purpose | Images | Classes |
|---------|---------|--------|---------|
| RFMiD v1 | Train / Validate / Test | 3,200 (1920/640/640) | 45 |
| IDRiD | LAM segmentation training | 54 train / 27 test | 5 lesion types |
| RFMiD 2.0 | External validation | 860 | — |
| MuReD | External validation | 2,208 | 20 |
 
All datasets are publicly available. Download links:
- RFMiD v1: https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid
- IDRiD: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
- RFMiD 2.0: https://ieee-dataport.org/competitions/retinal-fundus-multi-disease-image-dataset-rfmid-20
- MuReD: https://github.com/cc-by-nc/MuReD
 
---
 
## Training Configuration
 
```python
# Stage 2 — LAMRetNet
backbone_lr    = 2e-5
head_lr        = 2e-4
weight_decay   = 1e-4
warmup_epochs  = 3
scheduler      = CosineAnnealingWarmRestarts(T0=30)
total_epochs   = 60
patience       = 15
batch_size     = 32
seed           = 42
precision      = AMP FP16
cv_strategy    = StratifiedKFold(n_splits=5, stratify=Disease_Risk)
tta_views      = 3 (original, H-flip, V-flip)
threshold      = 0.46
```
 
 
## Requirements
 
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
Pillow>=9.5.0
albumentations>=1.3.0
```
 
Install with:
```bash
pip install -r requirements.txt
```
 
---
 
## Usage
 
The full training pipeline is implemented as a Kaggle notebook.
 
**Run on Kaggle:**
1. Open the public notebook: [INSERT KAGGLE URL HERE]
2. Fork the notebook to your account
3. Attach the RFMiD v1 and IDRiD datasets
4. Run all cells in order
 
**Notebook structure:**
- Cell 1–3: Dataset loading and preprocessing
- Cell 4–6: Stage 1 — LAM training and evaluation
- Cell 7–12: Stage 2 — LAMRetNet 5-fold CV training
- Cell 13–15: Test set evaluation + TTA
- Cell 16–18: External validation (RFMiD 2.0, MuReD)
- Cell 19–21: Explainability (Grad-CAM++, LayerCAM, MC Dropout)
 
---
 
## Explainability
 
LAMRetNet provides three complementary explainability outputs:
 
| Method | What it shows |
|--------|---------------|
| LAM attention map | Lesion-guided spatial prior (segmentation-derived) |
| Grad-CAM++ | Class-discriminative saliency; IoU vs IDRiD GT reported |
| LayerCAM | Multi-resolution hierarchical saliency |
| MC Dropout (N=20) | Predictive uncertainty: mean σ = 0.0326 per image |
 
---
 
 
## Acknowledgements
 
The authors thank the creators of the RFMiD, IDRiD, RFMiD 2.0, and MuReD datasets
for making their data publicly available to the research community.
 
