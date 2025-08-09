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
