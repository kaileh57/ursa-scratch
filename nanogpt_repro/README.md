# nanoGPT-from-scratch

This repository contains a from-scratch implementation of a GPT-2 model, along with all the scripts needed for training and evaluation. This project is based on the "Let's build GPT-2" video lecture.

## Quickstart Guide

This guide will walk you through setting up the environment, training the model, and evaluating it. All scripts are configured to be CPU-friendly.

### 1. Setup

First, you need to install the required Python packages. It is recommended to use a virtual environment.

```bash
pip install torch numpy transformers datasets tiktoken tqdm
```
If you want to use the jupyter notebook `play.ipynb` you will also need `matplotlib` and `jupyter`:
```bash
pip install matplotlib jupyter
```

### 2. Training

The main training script is `train_gpt2.py`.

#### TinyShakespeare (for testing)

A small dataset called `input.txt` (Tiny Shakespeare) is included for you to quickly test the training setup.

To start training on Tiny Shakespeare, simply run:

```bash
python train_gpt2.py
```

This will start training a small GPT-2 model on your CPU. The script is configured to automatically detect and use the CPU if a CUDA-enabled GPU is not available. You will see log outputs for training loss. This is a good way to verify that everything is working correctly.

#### FineWeb-Edu (for a full training run)

For a more serious training run, you can use the FineWeb-Edu dataset, a large, high-quality dataset of educational content.

**a. Download and Prepare the Data**

The `fineweb.py` script will download the 10-billion token sample of the FineWeb-Edu dataset and pre-tokenize it into shards. This is a large dataset, and this process will take a significant amount of time and disk space (around 20GB).

Run the script from within the `nanogpt_repro` directory:

```bash
python fineweb.py
```

This will create a directory named `edu_fineweb10B` containing the data shards.

**b. Start Training**

Once the dataset is ready, you can start the main training process. The `train_gpt2.py` script is pre-configured to use this dataset if the `edu_fineweb10B` directory exists.

```bash
python train_gpt2.py
```

This will train the 124M parameter GPT-2 model. On a modern CPU, this will be a very long process. For reference, the video mentions this taking about 1.7 hours on 8 A100 GPUs. On a CPU, expect this to take orders of magnitude longer. The script will save checkpoints and a log file (`log/log.txt`) with training progress.

### 3. Evaluation

The `hellaswag.py` script is provided to evaluate the model on the HellaSwag benchmark, which tests commonsense reasoning.

To evaluate the pretrained GPT-2 model from Hugging Face (e.g., 'gpt2'), run:

```bash
python hellaswag.py --device cpu
```

The training script `train_gpt2.py` also includes logic to periodically evaluate on HellaSwag during training.

### 4. Reference Implementation

This repository also includes `reference.c`, which is a C-language implementation of GPT-2. Please note that this is provided for reference purposes and is based on a CUDA implementation. It is not designed to be compiled and run on a CPU without significant modifications. The primary, runnable implementation for your use case is the Python code.

### 5. Code Overview

*   `train_gpt2.py`: Main script for training and evaluation.
*   `fineweb.py`: Script to download and process the FineWeb-Edu dataset.
*   `hellaswag.py`: Script for HellaSwag evaluation.
*   `reference.c`: Reference C/CUDA implementation of GPT-2.
*   `input.txt`: TinyShakespeare dataset for quick testing.
*   `play.ipynb`: A Jupyter Notebook to experiment with the model. 