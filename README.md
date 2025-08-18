<div align="center">
<h1>PTQ for ViM </h1>
<h3>PTQ for ViM with k-Scaled Quantization and Reparameterization</h3>

Bo-Yun Shi, Yi-Cheng Lo, Yi-Min Tsai, and An-Yeu (Andy) Wu

Paper: ([arXiv 2501.16738](https://arxiv.org/abs/2501.16738))

</div>

## Abstract
In this work, we focus on the post-training quantization (PTQ) of Vision Mamba. We address the issues with three core techniques: 1) a k-scaled token-wise quantization method for linear and convolutional layers, 2) a reparameterization technique to simplify hidden state quantization, and 3) a factor-determining method that reduces computational overhead by integrating operations. Through these methods, the error caused by PTQ can be mitigated. Experimental results on ImageNet-1k demonstrate only a 0.8–1.2\% accuracy degradation due to PTQ, highlighting the effectiveness of our approach.


## Getting Started

### Create Environment (using conda)

```bash
conda create -n {env_name} python=3.10.3
conda activate {env_name}
```

### Install torch + cuda toolkit

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
```

### Download ViM Files from GitHub

Clone the repository:

```bash
git clone https://github.com/Access-IC-Lab/PTQ_for_ViM.git
cd PTQ_for_ViM
```

Alternatively, manually download the zip file from:  
[https://github.com/Access-IC-Lab/PTQ_for_ViM](https://github.com/Access-IC-Lab/PTQ_for_ViM)

### Install Required Packages

```bash
pip install -r requirements.txt
```

## Inference Model

### Running Quantization and Inference

To run calibration for quantization and inference the quantized model:
```shell
python main.py --eval --resume {path_to_pretrained_model.pth} --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path {path_to_dataset} --device cuda --batch-size 128 --quantization-config PTQ4ViM --calibration-size 256 --quantization
```

### Running Using Scripts

Run FP inference on ViM tiny/small model on a specific GPU (replace `{t/s}` with `t` or `s`, and `{GPU}` with the GPU number):

```bash
source scripts/eval-pt-{t/s}.sh {GPU}
```

Run quantization on ViM tiny/small model on a specific GPU:

```bash
source scripts/q-eval-pt-{t/s}.sh {GPU}
```

### Quantization Configuration Setting

QuantConfig class can be found in `quant_configs/QuantConfig.py`. Basic setting for this work is defined in `quant_configs/PTQ4ViM.py`.


## Acknowledgement

This project is based on Mamba([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Vision Mamba([paper](https://arxiv.org/abs/2401.09417), [code](https://github.com/hustvl/Vim)), and the quantization framework is partially adopted from PTQ4ViT([code](https://github.com/hahnyuan/PTQ4ViT)). Thanks for their excellent works.



# Multi-Distillation from Speech and Music Representation Models

This repository contains the official implementation for our ASRU 2025 submission:  
**Multi-Distillation from Speech and Music Representation Models**.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Datasets & Teacher Models](#datasets--teacher-models)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

Real-world audio often mixes speech and music, but most models process only one domain.  
We propose a multi-teacher distillation framework to unify speech and music models into a single compact model, leveraging HuBERT/WavLM (for speech) and MERT (for music) as teachers.  
Our model performs well on both domains and shows improved few-shot generalization.

---

## Features

- Multi-teacher distillation (speech + music SSL models)
- Data-domain separation and hybrid feature translators
- Adaptive loss weighting for balanced learning
- Comprehensive evaluation on speech and music benchmarks (SUPERB, MARBLE)
- Few-shot learning experiments

---

## Getting Started

1. **Install Dependencies**  
   (Provide `requirements.txt` or list main packages)

2. **Download Teacher Model Checkpoints**
   - [HuBERT-base (ls960)](https://huggingface.co/facebook/hubert-base-ls960)
   - [WavLM-base+](https://huggingface.co/microsoft/wavlm-base-plus)
   - [MERT-public-v0](https://huggingface.co/m-a-p/MERT-v0-public)

3. **Prepare Data**
   - [LibriSpeech](https://www.openslr.org/12)
   - [Music4All](https://sites.google.com/view/contact4music4all)

4. **Implementation**
   
   Multi-Distillation related code can be found in `s3prl/s3prl/pretrain/multi_distiller`.

   - **Change to code directory:**  
     ```bash
     cd s3prl/s3prl
     ```

   - **Pretrain (multi-teacher distillation):**  
     ```bash
     python run_pretrain.py -u multi_distiller \
       -g config_model.yaml \
       -n exp_name
     ```
     - `-u`: upstream distiller name  
     - `-g`: config yaml file  
     - `-n`: experiment name  

   - **Downstream training / evaluation:**  
     ```bash
     python run_downstream.py -m train \
       -u multi_distiller_local \
       -d speech_commands \
       -s states-epoch-12.ckpt \
       -g config_model.yaml \
       -n exp_name \
       -o "config.downstream_expert.datarc.speech_commands_root=/path/to/your/speech_commands/"
     ```
     - `-m`: mode (`train` for training)
     - `-u`: upstream setting
     - `-d`: downstream task name
     - `-s`: pretrained checkpoint path
     - `-g`: config yaml for the checkpoint
     - `-n`: experiment name
     - `-o`: override (e.g., dataset path)

   > For more details on arguments and usage, please refer to the [s3prl repository](https://github.com/s3prl/s3prl) and its documentation.
