# BEACON: Blockchain Edge-based Anomaly Detection with Consensus Over Networks

BEACON is a distributed framework designed for blockchain anomaly detection in decentralized networks, focusing on edge-based processing, Byzantine fault tolerance, and cross-chain communication. The framework employs real-time network condition monitoring to maintain high performance across varying network topologies and resource constraints.

This repository contains the source code and experimental setup for **BEACON**. The framework addresses critical challenges in blockchain security, including the detection of illicit transactions, cross-chain money laundering, and Byzantine adversarial behavior.

## Table of Contents

* [Introduction](#introduction)
* [System Requirements](#system-requirements)
* [Key Features](#key-features)
* [Installation and Setup](#installation-and-setup)
* [Usage](#usage)

  * [Training the Model](#training-the-model)
  * [Running Experiments](#running-experiments)
  * [Distributed Setup](#distributed-setup)
* [Directory Structure](#directory-structure)

## Introduction

Blockchain technologies have introduced decentralized financial systems but have also exposed critical vulnerabilities. Malicious actors exploit these systems through cross-chain attacks, money laundering schemes, and evasion of centralized detection mechanisms. BEACON addresses these challenges by integrating the following components:

* **Edge-based anomaly detection**: Deploys lightweight detection models on resource-constrained devices.
* **Byzantine-robust consensus aggregation**: Ensures robust detection even in the presence of malicious nodes.
* **Cross-chain synchronization**: Detects illicit behavior across multiple blockchain networks.
* **Network-aware processing**: Adapts to real-time network conditions to ensure detection quality under varying latencies and packet losses.

The system is scalable, fault-tolerant, and optimized for real-time anomaly detection in distributed blockchain environments.

## System Requirements

* **Hardware**:

  * GPUs with CUDA support (optional but recommended for training).
  * At least 32GB of RAM for running experiments; 384GB RAM is optimal for large-scale tests.
* **Software**:

  * Python 3.7+ (with package management via `pip` or `conda`).
  * PyTorch (version 1.10 or later) with GPU support.
  * Other dependencies listed in `requirements.txt`.

## Key Features

1. **Edge-based Anomaly Detection**:

   * Utilizes distributed edge nodes for anomaly detection in blockchain transactions, enabling low-latency and scalable anomaly detection.

2. **Byzantine Fault Tolerance**:

   * Ensures robust detection against up to one-third of the nodes performing malicious actions (e.g., model poisoning, data poisoning).

3. **Cross-Chain Synchronization**:

   * Synchronizes detection results across different blockchain networks to detect attacks that span multiple chains.

4. **Network-Aware Processing**:

   * Integrates real-time network conditions such as latency, packet loss, and bandwidth utilization into the anomaly detection process.

5. **Federated Learning Support**:

   * Federated learning model training with consensus aggregation, ensuring that all participants (nodes) contribute to model training without exposing sensitive data.

6. **Scalability**:

   * BEACON is optimized to handle large-scale blockchain environments, supporting thousands of transactions per second.

## Installation and Setup

1. **Clone the Repository**:

  

2. **Create a Virtual Environment (Optional but Recommended)**:

   
   python3 -m venv beacon-env
   source beacon-env/bin/activate  # On Windows use `beacon-env\Scripts\activate`
   

3. **Install Dependencies**:

   Install the required Python packages using `pip`:

  
   pip install -r requirements.txt
   

4. **Install CUDA Drivers** (if using GPU):

   * Follow the instructions for your specific platform to install CUDA. Ensure that `torch.cuda.is_available()` returns `True`.

## Usage

### Training the Model

To train the BEACON model on a blockchain dataset, you need to configure the model, set training parameters, and run the training script.

1. **Configure the Experiment**:

   Modify the `config.yaml` file to set your desired parameters (e.g., number of edge nodes, training epochs, learning rate).

2. **Train the Model**:

   
   python beacon_main.py --config config.yaml --train
   

   The training process will involve:

   * Data loading and preprocessing.
   * Training the model using federated learning or a single-node setup.
   * Consensus aggregation across distributed nodes.

### Running Experiments

To evaluate the performance of the model in different scenarios, use the experiment manager module. This will allow you to run distributed experiments, benchmark performance, and analyze results.

python beacon_experiments.py --config config.yaml --run-experiments


### Distributed Setup

BEACON supports **distributed training** and **federated learning**. The distributed setup allows running experiments across multiple GPUs or nodes in a cluster.

1. **Launch Distributed Training**:

   Use the following command to initiate distributed training:

 
   python beacon_main.py --config config.yaml --distributed
   

   This will initiate the process of training the model across multiple nodes, ensuring that the distributed training mechanism (via PyTorch DDP) is utilized.

2. **Federated Learning Setup**:

   The framework supports federated learning for collaborative anomaly detection. Ensure that the `federated_learning` section in `config.yaml` is properly configured.

3. **GPU Support**:

   BEACON leverages GPUs to accelerate model training. Ensure that you have configured your system correctly to utilize GPU resources. If running on a multi-GPU setup, ensure the environment variable `CUDA_VISIBLE_DEVICES` is set properly.

## Directory Structure

/BEACON
|-- beacon_core.py              # Core model and data processing modules
|-- beacon_experiments.py       # Script for running experiments and benchmarking
|-- beacon_main.py              # Main entry point for training and evaluation
|-- beacon_protocols.py         # Networking protocols and cross-chain communication
|-- beacon_utils.py             # Utility functions for data loading and network simulation
|-- config.yaml                 # Configuration file for experiments and setup
|-- requirements.txt            # Python dependencies for the framework
|-- results/                    # Folder for storing experiment results
