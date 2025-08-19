# TFDC: Temporal-Frequency Enhanced Dual-path Complementary Contrastive Learning for Multivariate Time-Series Clustering

This repository contains the official implementation of the paper:  
**"TFDC: Multivariate Time-Series Clustering via Temporal-Frequency Enhanced Contrastive Learning"**

## 📖 Abstract

Multivariate Time-Series (MTS) clustering is a fundamental task in signal processing and data mining. Existing contrastive learning-based methods often suffer from two limitations:  
1. Ignoring clustering structures when constructing positive/negative pairs.  
2. Introducing unreasonable inductive biases through aggressive augmentations that distort temporal dependencies.

To address these, we propose **TFDC**, a novel framework that integrates:
- A **temporal-frequency co-enhancement mechanism** to generate low-distortion augmented views.
- A **dual-path architecture** that jointly optimizes cluster structure and representation fidelity through pseudo-label guided contrastive learning and masked reconstruction.

Experiments on five real-world UEA datasets show that TFDC outperforms SOTA methods by an average of **12.58% in NMI**.

## 🚀 Features

- ✅ Temporal-frequency enhancement with adaptive neighbor mixing
- ✅ Dual-path learning: PGCL (contrastive) + READ (reconstruction)
- ✅ Cluster-aware high-confidence sampling for reliable contrastive pairs
- ✅ End-to-end self-supervised training
- ✅ Compatible with GPU acceleration

## Training
python train.py

## 📦 Installation

### Requirements

- Python 3.10+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU support)

### Install Dependencies

```bash
pip install torch torchvision scikit-learn scipy fastdtw tqdm numpy optuna



