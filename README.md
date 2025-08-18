<div align="center">
<h1>Multi-Distillation </h1>
<h3>Multi-Distillation from Speech and Music Representation Models</h3>

Jui-Chiang Wei, Yi-Cheng Lin, Fabian Ritter-Gutierrez, Hung-yi Lee

Paper: ([arXiv 2501.16738](https://arxiv.org/abs/2501.16738))
Paper: ([arXiv 2506.07237](https://arxiv.org/abs/2506.07237)

</div>

## Abstract
Real-world audio often mixes speech and music, yet models typically handle only one domain. This paper introduces a multi-teacher distillation framework that unifies speech and music models into a single one while significantly reducing model size. Our approach leverages the strengths of domain-specific teacher models, such as HuBERT for speech and MERT for music, and explores various strategies to balance both domains. Experiments across diverse tasks demonstrate that our model matches the performance of domain-specific models, showing the effectiveness of cross-domain distillation. Additionally, we conduct few-shot learning experiments, highlighting the need for general models in real-world scenarios where labeled data is limited. Our results show that our model not only performs on par with specialized models but also outperforms them in few-shot scenarios, proving that a cross-domain approach is essential and effective for diverse tasks with limited data.


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
