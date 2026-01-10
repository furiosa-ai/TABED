# TABED: Test-Time Adaptive Ensemble Drafting for Robust Speculative Decoding in LVLMs

<p align="center">
    <a href="https://scholar.google.com/citations?user=XJXKp60AAAAJ&hl=en" target="_blank">Minjae Lee</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=Q-ARWkwAAAAJ&hl=en" target="_blank">Wonjun Kang</a><sup>1</sup>,
    <a href="https://dblp.org/pid/331/2300.html" target="_blank">Byeongkeun Ahn</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=TSo70_YAAAAJ&hl=en" target="_blank">Christian Classen</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=G1EpeWYAAAAJ&hl=en" target="_blank">Kevin Galim</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=IXJcR1gAAAAJ&hl=en" target="_blank">Seunghyuk Oh</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=grTN4yQAAAAJ&hl=en" target="_blank">Minghao Yan</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=Oyy8aDMAAAAJ&hl=en" target="_blank">Hyung Il Koo</a><sup>1</sup>,
    <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a><sup>2,3</sup>
</p>
<p align="center">
    <sup>1</sup>FuriosaAI, <sup>2</sup>UW-Madison, <sup>3</sup>KRAFTON AI
</p>

<p align="center">
    <!-- TODO: Add project page link -->
    <!-- <a href="https://your-project-page.github.io/">
        <img alt="Project" src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue&logo=github-pages">
    </a> -->
    <!-- TODO: Add arXiv link when available -->
    <!-- <a href="https://arxiv.org/abs/XXXX.XXXXX">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg">
    </a> -->
</p>

<p align="center">
<img src="asset/figure.png" width="100%" height="auto">
</p>
<p align="center">Figure 1. <b>Overview of TABED. </b></p>

***

## 📝 Abstract

Speculative decoding (SD) has proven effective for accelerating LLM inference by quickly generating draft tokens and verifying them in parallel. However, SD remains largely unexplored for Large Vision-Language Models (LVLMs), which extend LLMs to process both image and text prompts. To address this gap, we benchmark existing inference methods with small draft models on 11 datasets across diverse input scenarios and observe scenario-specific performance fluctuations. Motivated by these findings, we propose **Test-time Adaptive Batched Ensemble Drafting (TABED)**, which dynamically ensembles multiple drafts obtained via batch inference by leveraging deviations from past ground truths available in the SD setting. The dynamic ensemble method achieves an average robust walltime speedup of 1.74× over autoregressive decoding and a 5% improvement over single drafting methods, while remaining training-free and keeping ensembling costs negligible through parameter sharing. With its plug-and-play compatibility, we further enhance TABED by integrating advanced verification and alternative drafting methods.

***

## 🔗 Checkpoints

In the [Google Drive link](https://drive.google.com/drive/folders/1VO5XB4piOBQH-nXk9q-nXAzVvtPjiHRY?usp=sharing), the following model checkpoints are released:

| Model | Parameters | Training method |
|-------|------------|------------|
| LLaVA-1.5-68m | 68M | LLaVA-1.5 |
| LLaVA-OV-68m | 68M | LLaVA-OneVision |
| LLaVA-1.5-160m | 160M | LLaVA-1.5 |
| LLaVA-1.5-290m | 290M | LLaVA-1.5 |

***

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)
- PyTorch 2.3.1+

### Installation

```bash
# Clone the repository
git clone https://github.com/furiosa-ai/TABED.git
cd TABED
```

#### Option 1: Using uv (Recommended - Fast)

[uv](https://github.com/astral-sh/uv) provides fast, reproducible environment setup.

```bash
# Run the install script (installs uv if needed)
./install.sh

# Activate the environment
source .venv/bin/activate
```

#### Option 2: Using conda

```bash
# Create and activate conda environment
conda create -n tabed python=3.10 -y
conda activate tabed

# Install PyTorch with CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Environment Configuration

```bash
# Set your Hugging Face token for model access
export HF_TOKEN="your_huggingface_token"
```

***


## 🚀 Running Experiments

### Drafting Methods

This repository supports various drafting configurations:
#### 1. Single Drafting Method
- M (Multimodal) Uses full multimodal input
- T (Text-only) Uses only text input
- C (Caption) Uses image captions
- P (Pool): Uses pooled image features

#### 2. Static Ensemble Drafting Method ([In-batch Ensemble Drafting; IbED](https://openreview.net/pdf?id=ffDhpmwqdu))

- MT
- MTC
- MTP
- MTCP

#### 3. Dynamic Ensemble Drafting Method (Test-Time Adaptive Ensemble Drafting; TABED)
- **TabedMT** 
- **TabedMTC** 
- **TabedMTP** 
- **TabedMTCP**

### Reproducing Paper Results

We provide two evaluation scripts for comprehensive benchmarking:
<p align="center">
<img src="asset/table1.png" width="100%" height="auto">
</p>
<p align="center"> Table 1. Block efficiency results for drafting methods.</p> 

```bash
# Single-turn experiments
bash run_scripts/first_ALL.sh

# Multi-turn experiments (only if the first-turn expts are run)
bash run_scripts/second_ALL.sh
```



***

## 🙏 Acknowledgements

This project builds upon the work of several open-source repositories:
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [llava-hf (huggingface)](https://huggingface.co/llava-hf)

***

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
# Todo
```

***

## 📋 TODO
- Add arXiv link and badge once paper is published
- Release batched inference rather than sequential inference