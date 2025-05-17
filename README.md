# FED-PRIME: Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data

This repository contains the official implementation of **FED-PRIME**, a novel federated prompt-tuning framework designed to handle **heterogeneous and incomplete multimodal data** in decentralized learning environments.

This work builds upon and extends Federated Learning concepts from the [easyFL](https://github.com/WwZzz/easyFL/tree/main) framework.  
**easyFL: A Lightning Framework for Federated Learning** , which provides the PyTorch implementation for the IJCAI'21 paper _Federated Learning with Fair Averaging_.


---

## Overview

FED-PRIME is a federated prompt-tuning framework designed to tackle the challenges of **heterogeneous and incomplete multimodal client data** common in real-world federated learning. Unlike existing methods, FED-PRIME handles both **inter-client heterogeneity** (different clients have varying missing modalities) and **intra-client heterogeneity** (missing patterns vary within client samples) in pretrained parameter-efficient fine-tuning settings.

The approach learns **two distinct prompt sets per client**:  
- **Inter-client prompts** capture shared missing modality patterns across clients, while  
- **Intra-client prompts** model client-specific, input-agnostic missingness.


Key strengths include robust handling of incomplete multimodal data, efficient fine-tuning without full model updates, and superior prompt alignment to boost federated learning performance under diverse missing data scenarios.

### Repository Structure
```bash
├── algorithm/                  # Server Aggregation Components
├── benchmark/                  # Clients' Model Architectures
├── notebook/                   # Data preprocess files
├── scripts/                    # Evaluation scripts
├── utils/                      # Utility files
├── README.md                   # Data readme files
├── env_setup.sh                # Set up environment to run
├── generate_fedtask.py         # Create data files (client data partition) before running FL
├── main.py                     # Main running files
├── README.md                   # This file
├── requirements.txt            # Required Python packages
└── sweep_config.yaml           # Auto hyperparameter tuning
```

---

## Installation

```bash
bash env_setup.sh
```
---
## Usage

### Dataset Preparation

Prepare the multimodal datasets with simulated missing data patterns as described in the paper (e.g., UPMC Food-101, MM-IMDB). Note that both datasets are processed to include only the **8 most frequent classes** to simulate smaller client datasets typical in federated learning. Preprocessing scripts are included in the `data` folder.

We use two vision and language datasets: [MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb), and [UPMC Food-101](https://visiir.isir.upmc.fr/explore). Please download the datasets by yourself. We use `pyarrow` to serialize the datasets. Please see `DATA.md` to organize the datasets, run the following script to create the pyarrow binary file:
```bash 
python notebook/make_arrow_food101.py
python notebook/make_arrow_imdb.py
```

### Training and Evaluating

Experiment details can be found in `Experiments.md`

Generate fedtask (client data partition):
```bash
bash script/food101/main_exp_20clients/miss_both/20cl.sh
```

```bash
bash script/food101/main_exp_20clients/miss_both/all_models.sh
```
This script will:
- Generates FL data splits for 20 clients by running 20cl.sh.

- Defines which algorithms to run (default is "Fed-Intra").

- For each algorithm, runs main.py with specified training settings (task details, model, learning rate, epochs, batch size, GPU, etc.).

- Logs results with Weights & Biases and prints progress messages.

Any other scripts can be found in `./script/`

### Automatic Hyperparameter Tuning with W&B Sweeps

To perform automatic hyperparameter tuning, use Weights & Biases (W&B) sweep functionality:

1. **Create and start the sweep**  
   Run the sweep initialization command without activating the environment:  
   ```bash
   wandb sweep sweep_config_full.yaml
   ```
This returns a sweep ID (e.g., `bhbfcjr1`).

**Run the sweep agent**  
Use the sweep ID to start the sweep agent that runs experiments with different hyperparameter configurations:

```bash
wandb agent entity_name/project_name/bhbfcjr1
```



### Results
FED-PRIME achieves state-of-the-art performance in multimodal federated learning with heterogeneous missing data, outperforming adapted baselines by up to 107% relative improvement on classification accuracy/F1 scores across extensive experiments.

### Key findings include:

Robustness to missing data with varying missing rates.

Superior embedding alignment across incomplete multimodal inputs.

Efficient training with reduced GPU memory usage compared to baseline methods.

For detailed experimental results and analysis, please refer to the paper and supplementary material.
