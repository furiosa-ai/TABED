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
<p align="center"><b>Figure 1.</b></p>

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
# Set up environment variables
source setup_env.sh

# Set your Hugging Face token for model access
export HF_TOKEN="your_huggingface_token"
```

***

## ⚡ Quickstart

```bash
# Run speculative decoding with TABED on a single dataset
CUDA_VISIBLE_DEVICES=0 python3 main.py with SpecDecoding HalfPrecision LlavaBenchInTheWildData \
    drf=mjbooo/lvlm68m-ov \
    tgt=llava-hf/llava-v1.6-vicuna-7b-hf \
    TabedMT \
    exp_title=quickstart
```

***

## 🚀 Running Experiments

### Drafting Methods

TABED supports various drafting configurations:
- **M (Multimodal)**: Uses full multimodal input
- **T (Text-only)**: Uses only text input
- **C (Caption)**: Uses image captions
- **P (Pool)**: Uses pooled image features

Ensemble combinations: `TabedMT`, `TabedMTC`, `TabedMTP`, `TabedMTCP`, etc.

### Evaluation

```bash
# Single drafting method (Multimodal)
CUDA_VISIBLE_DEVICES=0 python3 main.py with SpecDecoding HalfPrecision LlavaBenchInTheWildData \
    drf=mjbooo/lvlm68m-ov tgt=llava-hf/llava-v1.6-vicuna-7b-hf \
    MultimodalDraft exp_title=eval-M

# TABED with MT ensemble
CUDA_VISIBLE_DEVICES=0 python3 main.py with SpecDecoding HalfPrecision LlavaBenchInTheWildData \
    drf=mjbooo/lvlm68m-ov tgt=llava-hf/llava-v1.6-vicuna-7b-hf \
    TabedMT tabed_rule=mm-weight mm_weight_policy=1 exp_title=eval-MT

# TABED with history-dependent weighting (MT*)
CUDA_VISIBLE_DEVICES=0 python3 main.py with SpecDecoding HalfPrecision LlavaBenchInTheWildData \
    drf=mjbooo/lvlm68m-ov tgt=llava-hf/llava-v1.6-vicuna-7b-hf \
    TabedMT tabed_rule=mm-weight mm_weight_policy=1 \
    history_dependent=True history_window=0 history_item=kld \
    exp_title=eval-MT-history
```

### Reproducing Paper Results

We provide two evaluation scripts for comprehensive benchmarking:

```bash
# Single-turn experiments (standard VQA tasks)
bash run_scripts/first_ALL.sh

# Multi-turn experiments (conversational tasks)
bash run_scripts/multi_ALL.sh
```

#### Configuring the Scripts

Both scripts support customization via configuration variables at the top of each file:

```bash
# GPU device selection
device_num=0

# Draft and target models
drfs=(mjbooo/lvlm68m)
tgts=(llava-hf/llava-1.5-7b-hf)

# TABED configuration
tabed_rules=(mm-weight)
mm_weight_policys=(1)
```

### Supported Datasets

**Single-image:** `LlavaBenchInTheWildData`, `DocVQAData`, `PopeData`, `MMVetData`

**Multi-image:** `IEditData`, `MagicBrushData`, `SpotTheDiffData`, `PororoSVData`, `VISTData`

***

## 📊 Results

TABED achieves an average robust walltime speedup of **1.74x** over autoregressive decoding and a **5% improvement** over single drafting methods across 9 diverse datasets.

***

## 🙏 Acknowledgements

This project builds upon the work of several open-source repositories:
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Sacred](https://github.com/IDSIA/sacred)

***

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
# Todo
```

***

## 📋 TODO

### High Priority
- Add arXiv link and badge once paper is published
- Add project page link (if applicable)

### Optional Enhancements
- Add demo script (`demo.py`)
- Add detailed results table/figures from the paper
