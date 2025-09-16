# TFEC: Multivariate Time-Series Clustering via Temporal-Frequency Enhanced Contrastive Learning

This repository contains the official implementation of our paper:  
**"TFEC: Multivariate Time-Series Clustering via Temporal-Frequency Enhanced Contrastive Learning"**

## 📖 Abstract

Multivariate Time-Series (MTS) clustering is crucial for signal processing and data analysis. Although deep learning approaches, particularly those leveraging Contrastive Learning (CL), are prominent for MTS representation, existing CL-based models face two key limitations:  
1. Neglecting clustering information during positive/negative sample pair construction;  
2. Introducing unreasonable inductive biases through augmentation strategies that destroy time dependence and periodicity.

To address these, this paper proposes **TFEC** — a novel Temporal-Frequency Enhanced Contrastive Learning framework that integrates:
- A **temporal-frequency co-enhancement mechanism** generating low-distortion representations through aligned cropping and adaptive spectral mixing.
- A **synergistic dual-path learning architecture** combining pseudo-label guided contrastive learning (PGCL) and reconstruction adjustment (READ) to jointly improve cluster distribution and representation fidelity.

Extensive evaluations on six real-world benchmark UEA datasets demonstrate that TFEC achieves state-of-the-art performance, with an average improvement of **4.48% in NMI** over strong baselines.

## 🚀 Key Features

- **Dual-Domain Enhancement**: Preserves temporal continuity via aligned cropping and enriches features through adaptive frequency mixing.
- **Cluster-Aware Sampling**: Selects high-confidence intra-cluster samples to form reliable contrastive pairs.
- **Dual-Path Learning**: PGCL path improves cluster compactness and separation; READ path ensures representation fidelity via masked EME reconstruction.
- **End-to-End Self-Supervised Training**: Fully differentiable framework compatible with GPU acceleration.

## 📦 Installation

### Requirements

- Python 3.10+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU support)

### Install Dependencies

```bash
pip install torch torchvision scikit-learn scipy numpy tqdm