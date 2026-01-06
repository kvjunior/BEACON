# BEACON: Byzantine-Resilient Blockchain Anomaly Detection with Network-Aware Distributed Consensus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**ACM CCS 2026 Submission — Anonymous Artifact Repository**

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Reproduction Instructions](#reproduction-instructions)
  - [Detection Performance (Table 3)](#1-detection-performance-table-3)
  - [Statistical Significance (Table 4)](#2-statistical-significance-table-4)
  - [Scalability Analysis (Table 5)](#3-scalability-analysis-table-5)
  - [Cross-Chain Performance (Table 6)](#4-cross-chain-performance-table-6)
  - [Byzantine Robustness (Section 4.6)](#5-byzantine-robustness-section-46)
  - [Ablation Study (Table 7)](#6-ablation-study-table-7)
  - [Edge Deployment (Section 4.8)](#7-edge-deployment-section-48)
  - [Literature Comparison (Table 8)](#8-literature-comparison-table-8)
- [Configuration](#configuration)
- [Pre-trained Models](#pre-trained-models)
- [Expected Results](#expected-results)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

BEACON is a distributed framework for blockchain anomaly detection that addresses three fundamental challenges in adversarial multi-chain environments:

1. **Network Degradation**: Maintaining detection accuracy when network latency exceeds 100ms or packet loss surpasses 5%
2. **Byzantine Adversaries**: Tolerating up to one-third malicious detection nodes
3. **Cross-Chain Correlation**: Detecting laundering patterns spanning heterogeneous blockchain networks

This repository contains the complete implementation, datasets, pre-trained models, and evaluation scripts necessary to reproduce all experimental results presented in the paper.

## Key Contributions

| Component | Description | Key Result |
|-----------|-------------|------------|
| **NetworkAwareMGD** | Network-condition-aware graph attention mechanism | >94% F1 under 200ms latency |
| **Byzantine Consensus** | Multi-strategy aggregation with dynamic trust scoring | <1.4% degradation at 33% malicious |
| **Cross-Chain Sync** | Temporal alignment across heterogeneous blockchains | 57.9ms P50, 98.98% accuracy |

---

## Repository Structure

```
BEACON-CCS2026/
│
├── README.md                    # This file
├── LICENSE                      # MIT License
├── config.yaml                  # Global configuration parameters
├── requirements.txt             # Python dependencies
│
├── beacon_core.py               # Core model architectures
│   ├── NetworkAwareMGD          #   Network-aware graph attention (Section 3.2)
│   ├── DistributedEdge2Seq      #   Edge node sequence encoder
│   └── BEACONModel              #   Full model integration
│
├── beacon_protocols.py          # Distributed protocols
│   ├── ByzantineConsensus       #   Multi-strategy aggregation (Section 3.3)
│   ├── TrustScoring             #   Dynamic trust mechanism (Eq. 5)
│   └── CrossChainSync           #   Synchronization protocol (Section 3.4)
│
├── beacon_experiments.py        # Evaluation and benchmarking
│   ├── DetectionEvaluator       #   F1, Precision, Recall, AUC metrics
│   ├── ByzantineEvaluator       #   Attack simulation and robustness testing
│   ├── ScalabilityEvaluator     #   Node scaling experiments
│   └── CrossChainEvaluator      #   Multi-chain correlation tests
│
├── beacon_utils.py              # Utility functions
│   ├── DataLoader               #   Dataset loading and preprocessing
│   ├── NetworkSimulator         #   Network condition simulation
│   ├── MetricsLogger            #   Experiment logging
│   └── StatisticalTests         #   Wilcoxon, Cohen's d, Bonferroni
│
├── beacon_main.py               # Main entry point and CLI
│
├── Figures/                     # Generated figures for paper
│   ├── fig_main_results.png     #   Figure 3: Main experimental results
│   ├── fig_crosschain.png       #   Figure 4: Cross-chain analysis
│   └── beacon_architecture.*    #   Figure 2: Architecture diagram
│
├── data/                        # Datasets (see Datasets section)
│   ├── ethereum_s/              #   Ethereum-S: 1.33M addresses
│   ├── ethereum_p/              #   Ethereum-P: 5.67M addresses
│   ├── bitcoin_m/               #   Bitcoin-M: 8.91M addresses
│   └── bitcoin_l_sample/        #   Bitcoin-L: 10% stratified sample
│
├── models/                      # Pre-trained model weights
│   ├── beacon_eth_s.pt          #   Ethereum-S checkpoint
│   ├── beacon_eth_p.pt          #   Ethereum-P checkpoint
│   ├── beacon_btc_m.pt          #   Bitcoin-M checkpoint
│   └── beacon_btc_l.pt          #   Bitcoin-L checkpoint
│
├── configs/                     # Experiment configurations
│   ├── detection.yaml           #   Detection experiments
│   ├── byzantine.yaml           #   Byzantine robustness tests
│   ├── scalability.yaml         #   Scaling experiments
│   └── crosschain.yaml          #   Cross-chain evaluation
│
├── scripts/                     # Automation scripts
│   ├── train.py                 #   Model training
│   ├── evaluate.py              #   Full evaluation suite
│   └── reproduce_all.sh         #   Reproduce all results
│
└── results/                     # Raw experimental outputs
    ├── detection/               #   Table 3 results
    ├── significance/            #   Table 4 statistical tests
    ├── scalability/             #   Table 5 scaling data
    ├── crosschain/              #   Table 6 sync performance
    ├── byzantine/               #   Section 4.6 attack results
    ├── ablation/                #   Table 7 component analysis
    └── literature/              #   Table 8 comparison data
```

---

## Requirements

### Software Dependencies

- **Python**: 3.9 or higher
- **PyTorch**: 2.0 or higher (with CUDA 11.8+ for GPU acceleration)
- **PyTorch Geometric**: 2.4 or higher
- **Operating System**: Ubuntu 20.04+ or equivalent Linux distribution

### Python Packages

```
torch>=2.0.0
torch-geometric>=2.4.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.12.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.1
web3>=6.0.0
requests>=2.28.0
```

---

## Installation

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/BEACON-CCS2026/
cd BEACON-CCS2026

# Create conda environment
conda create -n beacon python=3.10 -y
conda activate beacon

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install remaining dependencies
pip install -r requirements.txt
```

### Option 2: Pip Installation

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/BEACON-CCS2026/
cd BEACON-CCS2026

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python beacon_main.py --version
```

Expected output:
```
PyTorch: 2.x.x, CUDA: True
PyG: 2.x.x
BEACON v1.0.0 - CCS 2026
```

---

## Datasets

### Included Datasets

| Dataset | Addresses | Transactions | Labels | Illicit % | Size | Status |
|---------|-----------|--------------|--------|-----------|------|--------|
| Ethereum-S | 1.33M | 6.79M | 150K | 4.21% | 892 MB | ✓ Full |
| Ethereum-P | 5.67M | 28.4M | 420K | 3.87% | 2.1 GB | ✓ Full |
| Bitcoin-M | 8.91M | 67.2M | 680K | 5.12% | 3.8 GB | ✓ Full |
| Bitcoin-L | 20.1M | 203M | 1.2M | 4.58% | 4.2 GB | ◐ 10% Sample |

### Data Sources

Ground-truth labels are derived from:
- [Etherscan Label Cloud](https://etherscan.io/labelcloud) — Ethereum address classifications
- [WalletExplorer](https://www.walletexplorer.com/) — Bitcoin wallet clustering
- OFAC Sanctions List — Regulatory designations (public records)
- FinCEN Advisories — Financial crime indicators (public records)

### Downloading Datasets

```bash
# Datasets are included in the repository
# Verify integrity after cloning:
python beacon_main.py --verify-data

# For Bitcoin-L full dataset (180 GB), contact authors or download from:
# Instructions provided in data/bitcoin_l_sample/FULL_DATASET.md
```

### Data Format

Each dataset directory contains:
```
dataset_name/
├── addresses.csv          # Node features (8 dimensions per address)
├── transactions.csv       # Edge list with transaction attributes
├── labels.csv             # Ground-truth: 0 = legitimate, 1 = illicit
├── splits/                # Train/val/test splits (5 seeds)
│   ├── seed_42/
│   ├── seed_123/
│   ├── seed_456/
│   ├── seed_789/
│   └── seed_1011/
└── metadata.json          # Dataset statistics and provenance
```

### Node Features (8 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `tx_count` | Total transaction count |
| 1 | `total_value` | Cumulative value transferred |
| 2 | `avg_value` | Average transaction value |
| 3 | `in_degree` | Number of incoming transactions |
| 4 | `out_degree` | Number of outgoing transactions |
| 5 | `clustering` | Local clustering coefficient |
| 6 | `temporal_var` | Temporal activity variance |
| 7 | `counterparty_count` | Unique counterparty addresses |

---

## Quick Start

### Training a Model

```bash
# Train on Ethereum-S dataset
python beacon_main.py train \
    --dataset ethereum_s \
    --config configs/detection.yaml \
    --output models/beacon_eth_s.pt

# Train with specific seed for reproducibility
python beacon_main.py train \
    --dataset ethereum_s \
    --seed 42 \
    --epochs 100 \
    --patience 10
```

### Evaluation with Pre-trained Model

```bash
# Evaluate pre-trained model on Ethereum-S
python beacon_main.py evaluate \
    --dataset ethereum_s \
    --checkpoint models/beacon_eth_s.pt \
    --metrics f1,precision,recall,auc
```

### Running All Experiments

```bash
# Reproduce all paper results (approximately 18 hours on RTX 3090)
bash scripts/reproduce_all.sh

# Or run specific experiment
python beacon_main.py experiment --name detection --all-datasets
```

---

## Reproduction Instructions

This section provides commands to reproduce each table and figure in the paper. All experiments use 5 random seeds `{42, 123, 456, 789, 1011}` and report mean ± standard deviation.

### 1. Detection Performance (Table 3)

Reproduces F1, Precision, Recall, and AUC across all datasets and methods.

```bash
python beacon_main.py experiment \
    --name detection \
    --datasets ethereum_s,ethereum_p,bitcoin_m,bitcoin_l \
    --methods beacon,gcn,graphsage,gat,single_chain,fedavg,krum_only \
    --seeds 42,123,456,789,1011 \
    --output results/detection/

# Expected runtime: ~4 hours on single RTX 3090
```

**Expected Results:**

| Dataset | BEACON F1 | Best Baseline F1 | Improvement |
|---------|-----------|------------------|-------------|
| Ethereum-S | 97.94 ± 0.23% | 91.45% (Krum) | +6.49% |
| Bitcoin-M | 94.26 ± 0.28% | 88.12% (Krum) | +6.14% |
| Bitcoin-L | 98.11 ± 0.19% | 92.34% (Krum) | +5.77% |

### 2. Statistical Significance (Table 4)

Computes Wilcoxon signed-rank tests and Cohen's d effect sizes.

```bash
python beacon_main.py experiment \
    --name significance \
    --input results/detection/ \
    --output results/significance/

# Expected runtime: ~5 minutes
```

**Expected Results:**
- All p-values < 0.001 after Bonferroni correction (24 comparisons)
- All Cohen's d > 2.4 (very large effect)

### 3. Scalability Analysis (Table 5)

Tests performance degradation across node counts from 100 to 5,000.

```bash
python beacon_main.py experiment \
    --name scalability \
    --nodes 100,500,1000,1500,2000,3000,5000 \
    --dataset ethereum_s \
    --output results/scalability/

# Expected runtime: ~3 hours
# Note: 5000 nodes requires >384GB RAM and will report OOM
```

**Expected Results:**

| Nodes | F1 (%) | Consensus (s) | Memory (GB) | Status |
|-------|--------|---------------|-------------|--------|
| 100 | 98.11 | 0.42 | 24 | ✓ Stable |
| 500 | 97.34 | 0.89 | 67 | ✓ Stable |
| 1,000 | 96.45 | 1.28 | 142 | ✓ Stable |
| 1,500 | 95.62 | 2.16 | 231 | ◦ Degraded |
| 2,000 | 94.12 | 3.45 | 298 | ◦ Degraded |
| 3,000 | 91.78 | 6.23 | 356 | ✗ Critical |
| 5,000 | — | — | >384 | ✗ OOM |

### 4. Cross-Chain Performance (Table 6)

Evaluates synchronization across six blockchain pairs.

```bash
python beacon_main.py experiment \
    --name crosschain \
    --chain-pairs eth-polygon,eth-bsc,eth-btc,btc-polygon,btc-bsc,bsc-polygon \
    --output results/crosschain/

# Expected runtime: ~2 hours
```

**Expected Results:**

| Chain Pair | Consensus Types | P50 (ms) | P95 (ms) | Accuracy (%) |
|------------|-----------------|----------|----------|--------------|
| ETH-Polygon | PoS-PoS | 48.7 | 67.2 | 99.43 |
| ETH-BSC | PoS-PoSA | 54.2 | 71.8 | 99.12 |
| ETH-BTC | PoS-PoW | 61.3 | 78.4 | 98.67 |
| BTC-Polygon | PoW-PoS | 63.8 | 79.1 | 98.71 |
| BTC-BSC | PoW-PoSA | 58.9 | 74.6 | 98.89 |
| BSC-Polygon | PoSA-PoS | 52.1 | 69.3 | 99.08 |
| **Average** | — | **56.5** | **73.4** | **98.98** |

### 5. Byzantine Robustness (Section 4.6)

Simulates four attack types with 33% malicious nodes.

```bash
python beacon_main.py experiment \
    --name byzantine \
    --attacks model_poisoning,data_poisoning,sybil_eclipse,delay \
    --malicious-fraction 0.33 \
    --methods beacon,fedavg,krum_only \
    --output results/byzantine/

# Expected runtime: ~4 hours
```

**Expected Results:**

| Attack Type | BEACON Degradation | FedAvg Degradation | Krum-Only Degradation |
|-------------|--------------------|--------------------|----------------------|
| Model Poisoning | 0.89% | 17.46% | 2.87% |
| Data Poisoning | 1.20% | 15.23% | 3.15% |
| Sybil-Eclipse | 1.40% | 18.12% | 4.23% |
| Delay | 0.70% | 12.34% | 1.92% |

### 6. Ablation Study (Table 7)

Isolates contribution of each component.

```bash
python beacon_main.py experiment \
    --name ablation \
    --components full,-network_aware,-crosschain,-byzantine,-adaptive \
    --dataset bitcoin_l \
    --output results/ablation/

# Expected runtime: ~2 hours
```

**Expected Results:**

| Configuration | F1 (%) | Δ vs Full |
|---------------|--------|-----------|
| Full BEACON | 98.11 | — |
| − NetworkAwareMGD | 94.86 | −3.25% |
| − Cross-Chain Sync | 95.31 | −2.80% |
| − Byzantine Consensus | 96.53 | −1.58% |
| − Adaptive Routing | 97.10 | −1.01% |
| Vanilla GAT (baseline) | 90.17 | −7.94% |

### 7. Edge Deployment (Section 4.8)

Profiles performance on resource-constrained hardware.

```bash
# Requires NVIDIA Jetson AGX Xavier or simulation mode
python beacon_main.py experiment \
    --name edge \
    --hardware jetson_agx \
    --quantization int8 \
    --output results/edge/

# Simulation mode (for reviewers without Jetson hardware):
python beacon_main.py experiment \
    --name edge \
    --simulate-edge \
    --output results/edge/
```

**Expected Results:**

| Dataset | F1 (%) | Throughput (tx/s) | Model Size | Energy (J/tx) |
|---------|--------|-------------------|------------|---------------|
| Ethereum-S | 94.8 | 1,210 | 42 MB | 0.94 |
| Ethereum-P | 93.2 | 1,150 | 42 MB | 1.02 |
| Bitcoin-M | 91.7 | 1,070 | 42 MB | 1.19 |
| Bitcoin-L | 92.4 | 1,130 | 42 MB | 1.08 |

### 8. Literature Comparison (Table 8)

Compares against four recent methods.

```bash
python beacon_main.py experiment \
    --name literature \
    --methods beacon,voronov2024,song2025,tharani2024,chen2025 \
    --datasets ethereum_s,bitcoin_m \
    --output results/literature/

# Expected runtime: ~3 hours
```

**Expected Results:**

| Method | Venue | F1 (%) | Latency | Throughput | Byzantine | Cross-Chain |
|--------|-------|--------|---------|------------|-----------|-------------|
| Voronov et al. | ToN'24 | 87.3 | 340ms | 2.1k tx/s | ✗ | ✗ |
| Song et al. | TIFS'25 | 91.2 | 180ms | 4.8k tx/s | ✗ | ✗ |
| Tharani et al. | TIFS'24 | 89.7 | 220ms | 3.8k tx/s | ✗ | ◦ |
| Chen et al. | INFOCOM'25 | 88.4 | 280ms | 3.2k tx/s | ◦ | ✗ |
| **BEACON** | — | **94.26–98.11** | **42–51ms** | **9.9–11.3k tx/s** | **✓** | **✓** |

### Generate All Figures

```bash
# Generate Figure 3 (main results - 4 panels)
python beacon_main.py visualize \
    --input results/ \
    --output Figures/fig_main_results.png \
    --figure main_results

# Generate Figure 4 (cross-chain and literature - 2 panels)
python beacon_main.py visualize \
    --input results/ \
    --output Figures/fig_crosschain_literature.png \
    --figure crosschain_literature

# Generate all figures at once
python beacon_main.py visualize --all --input results/ --output Figures/
```

---

## Configuration

### Global Configuration (`config.yaml`)

```yaml
# Model Architecture
model:
  hidden_dim: 256
  num_layers: 3
  num_heads: 8
  dropout: 0.1
  network_aware: true

# Training
training:
  epochs: 100
  batch_size: 1024
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 10
  scheduler: cosine_warmup
  mixed_precision: true
  gradient_clip: 1.0

# Byzantine Consensus
byzantine:
  trust_momentum: 0.9              # β in Eq. 5
  trust_weights: [0.4, 0.35, 0.25] # ω₁, ω₂, ω₃ for consistency, similarity, reliability
  detection_threshold: 2.0         # θ for outlier detection
  strategies: [krum, trimmed_mean, coord_median, weighted]

# Cross-Chain Synchronization
crosschain:
  correlation_threshold: 0.7       # θ_R in Eq. 8
  max_lag_blocks: 100
  finality:
    ethereum: 32
    bitcoin: 6
    bsc: 15
    polygon: 256

# Network Simulation
network:
  latency_range: [10, 300]         # milliseconds
  packet_loss_range: [0, 0.15]     # 0-15%
  bandwidth_min: 1.0               # Mbps
  jitter_max: 50                   # milliseconds

# Evaluation
evaluation:
  seeds: [42, 123, 456, 789, 1011]
  metrics: [f1, precision, recall, auc, fpr]
  confidence_level: 0.95
```

### Experiment-Specific Configurations

Located in `configs/` directory:

| File | Description | Key Parameters |
|------|-------------|----------------|
| `detection.yaml` | Detection performance | datasets, methods, seeds |
| `byzantine.yaml` | Byzantine robustness | attack types, malicious fraction |
| `scalability.yaml` | Node scaling | node counts, memory limits |
| `crosschain.yaml` | Cross-chain evaluation | chain pairs, consensus types |
| `edge.yaml` | Edge deployment | quantization, hardware profile |

---

## Pre-trained Models

Pre-trained model checkpoints are provided for immediate evaluation:

| Model | Dataset | F1 (%) | Size | SHA256 (first 8 chars) |
|-------|---------|--------|------|------------------------|
| `beacon_eth_s.pt` | Ethereum-S | 97.94 | 42 MB | `a3f8b2c1` |
| `beacon_eth_p.pt` | Ethereum-P | 96.78 | 42 MB | `e7d4a9f2` |
| `beacon_btc_m.pt` | Bitcoin-M | 94.26 | 42 MB | `c1b5e8d3` |
| `beacon_btc_l.pt` | Bitcoin-L | 98.11 | 42 MB | `f9a2c7b4` |

### Loading Pre-trained Models

```python
from beacon_core import BEACONModel
import torch

# Load model
model = BEACONModel.from_pretrained('models/beacon_eth_s.pt')
model.eval()

# Inference
with torch.no_grad():
    predictions = model(graph_data, network_state)
```

### Verifying Model Integrity

```bash
# Verify checksums
python beacon_main.py --verify-models

# Expected output:
# ✓ beacon_eth_s.pt: a3f8b2c1... (VALID)
# ✓ beacon_eth_p.pt: e7d4a9f2... (VALID)
# ✓ beacon_btc_m.pt: c1b5e8d3... (VALID)
# ✓ beacon_btc_l.pt: f9a2c7b4... (VALID)
```

---

## Expected Results

### Summary of Key Metrics

| Metric | Paper Claim | Expected Range | Tolerance |
|--------|-------------|----------------|-----------|
| F1-Score (Ethereum-S) | 97.94% | 97.71 – 98.17% | ±0.23% |
| F1-Score (Bitcoin-L) | 98.11% | 97.92 – 98.30% | ±0.19% |
| Byzantine Degradation (33%) | 0.89% | 0.75 – 1.10% | ±0.21% |
| Cross-Chain Latency (P50) | 57.9 ms | 55.2 – 60.6 ms | ±2.7 ms |
| Cross-Chain Accuracy | 98.98% | 98.82 – 99.14% | ±0.16% |
| Throughput | 10,610 tx/s | 9,900 – 11,300 tx/s | ±7% |

**Note:** Small variations from reported values are expected due to:
- Hardware differences (GPU architecture, memory bandwidth)
- Random initialization and non-deterministic CUDA operations
- Floating-point precision variations

Results within the expected range constitute successful reproduction.

---

## Hardware Requirements

### Minimum Requirements (Evaluation Only)

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GPU with 8GB+ VRAM (RTX 2080 or equivalent) |
| RAM | 32 GB |
| Storage | 50 GB free space |
| CUDA | 11.8 or higher |

### Recommended Requirements (Full Reproduction)

| Component | Specification |
|-----------|---------------|
| GPU | 4× NVIDIA RTX 3090 (24GB each) |
| RAM | 384 GB (required for scalability experiments >1,500 nodes) |
| Storage | 200 GB SSD |
| CUDA | 11.8 or higher |
| CPU | Intel Xeon Silver 4314 (64 cores) or equivalent |

### Edge Deployment Hardware

| Device | Specification |
|--------|---------------|
| Primary | NVIDIA Jetson AGX Xavier (32GB) |
| Alternative | NVIDIA Jetson Orin NX (16GB) with reduced batch size |

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Reduce batch size
python beacon_main.py train --batch-size 512

# Enable gradient checkpointing
python beacon_main.py train --gradient-checkpoint

# Use mixed precision training
python beacon_main.py train --mixed-precision
```

**2. PyTorch Geometric Installation Failures**

```bash
# Install with specific CUDA version
pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**3. Dataset Verification Errors**

```bash
# Re-download corrupted files
python beacon_main.py --verify-data --redownload

# Check dataset integrity
python beacon_main.py --check-checksums
```

**4. Reproducibility Variations**

```bash
# Enable full determinism (slower training)
python beacon_main.py train --deterministic --seed 42

# Set environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
```

**5. Memory Issues with Large Graphs**

```bash
# Use mini-batch training
python beacon_main.py train --mini-batch --num-neighbors 15,10

# Enable CPU offloading
python beacon_main.py train --cpu-offload
```

### Performance Debugging

```bash
# Profile GPU utilization
python beacon_main.py train --profile --profile-output profile.json

# Monitor memory usage
python beacon_main.py train --monitor-memory
```

### Getting Help

For issues not covered above:

1. Check existing issues in the repository
2. Search error messages in documentation
3. Open a new issue with:
   - Full error message and traceback
   - Hardware configuration (`nvidia-smi` output)
   - Python package versions (`pip freeze > versions.txt`)
   - Minimal steps to reproduce
   - Operating system and version

---

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Anonymous Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Dataset Licenses

| Dataset | Source | License |
|---------|--------|---------|
| Ethereum transactions | Public blockchain | No restrictions |
| Bitcoin transactions | Public blockchain | No restrictions |
| Etherscan labels | Etherscan.io | Attribution required |
| WalletExplorer labels | WalletExplorer.com | Attribution required |
| OFAC designations | U.S. Treasury | Public domain |

---

## Acknowledgments

*Acknowledgments withheld for anonymous review.*

---

<p align="center">
  <b>Repository maintained for ACM CCS 2026 double-blind review</b><br>
  <i>Last updated: January 2026</i>
</p>
