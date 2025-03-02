# TABED: Test-Time Adaptive Ensemble Drafting for Robust Speculative Decoding in LVLMs

<p align="center">
| <a href="https://arxiv.org/abs/XXXX"><b>Arxiv Paper</b></a> |
</p>

## Abstract

Speculative decoding (SD) has proven effective in accelerating LLM inference by swiftly generating draft tokens and verifying them in parallel.
However, SD remains largely unexplored for Large Vision Language Models (LVLMs), advanced LLMs capable of processing both image and text prompts.
To address this gap, we first benchmark existing drafting methods for LVLMs across diverse scenarios and observe that methods using small draft models show scenario-specific performance fluctuations.
Motivated by these findings, we propose Test-time Adaptive Batched Ensemble Draft (TABED), which dynamically ensembles multiple drafts obtained via batch inference by leveraging measurable deviations of the drafts from past ground truth available in SD setting.
Across diverse input scenarios, TABED achieves an average robust speedup of 1.8x and a 5\% improvement compared to individual draftings, though it does not incur additional costs during training (i.e., training-free) or inference.
To further enhance its extensibility, we also explore and incorporate alternative draftings using image pooling and captioning.
Our method maintains seamless compatibility with existing LVLM acceleration techniques, and we open-source custom-trained draft LVLMs to ensure reproducibility.

## Checkpoints
In the [Google-drive link](https://drive.google.com/drive/folders/1VO5XB4piOBQH-nXk9q-nXAzVvtPjiHRY?usp=sharing), the following model checkpoints are released:
- LLaVA-1.5-68m
- LLaVA-OV-68m
- LLaVA-1.5-160m
- LLaVA-1.5-290m

## Coming Soon
- Inference code for TABED for each configuration
