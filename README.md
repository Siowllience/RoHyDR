# RoHyDR: Robust Hybrid Diffusion Recovery for Incomplete Multimodal Emotion Recognition

[Introduction](#introduction)

[Quick Start](#quick-start)

[Note](#note)

## Introduction
We propose Robust Hybrid Diffusion Recovery (RoHyDR), a novel framework that performs missing-modality recovery at unimodal, multimodal, feature, and semantic levels.

### Motivation
 ![](figs/introduction.png)

 (1) Current research methods either exclusively focus on enhancing unimodal reconstruction capability or solely consider maintaining 
multimodal semantic consistency in the joint representation, resulting in suboptimal reconstruction of missing modalities and limited recovery in multimodal representation robustness. (2) The significant disparity in loss values between internal modules such as the generator and classifier continuously creates a gradient imbalance, degrading training stability and efficiency, as shown in (c).


### The Proposed RoHyDR Method

![](figs/framework.png)

In the context of IMER, multimodal data \([A; T; V]\) is categorized into two types: available modalities and missing modalities. Any single modality or any pair of modalities may be missing (vision modality is missed as an example here). During training, RoHyDR receives the ground-truth of the missing modalities, available modalities, and Gaussian noise as inputs. During testing, only the available modalities and Gaussian noise are provided as inputs.

#### Hybrid Recovery Method
(1) The unimodal representation recovery module incorporates a generator consisting of a high-dimensional diffusion model and a unimodal reconstructor to generate semantically aligned and distribution-consistent representations.(2) The multimodal fusion recovery module adopts adversarial learning to produce a realistic fused multimodal representation and recover missing semantic content, incorporating a fusion network, a multimodal reconstructor, and a discriminator.

#### Multi-Stage Optimization Strategy

The multi-stage optimization strategy is developed to effectively address the challenges in multi-objective function optimization, enabling systematic optimization of missing modality recovery, multimodal joint representation learning, and robust emotion classification.

## Quick Start

### Environments

- Python 3.8
- PyTorch 1.13.0 with CUDA 11.6

Please refer to `requirements.txt` for more information.


### Datasets
Due to the double-blind review process, we cannot provide a direct link to the CMU-MOSI and CMU-MOSEI datasets. You can access them via the CMU Multimodal Data SDK or obtain them from previously released open-source multimodal emotion recognition projects. 
Downloaded datasets can be put into the `dataset/` directory and organized as `dataset/MOSI/aligned_50.pkl` and `dataset/MOSEI/aligned_50.pkl`. Alternatively, you can modify the dataset path in `config/config.json`.



### Run the Code

```
python train.py
```

## Note

### Simple demo

In this version, we do not provide the full source code. Instead, we release a runnable simple demo intended to illustrate the model architecture and experimental settings:
- You can refer to `trains/mainNets/model/rohydr.py` for the proposed [Hybrid Recovery Method](#hybrid-recovery-method).
- The [Multi-Stage Optimization Strategy](#multi-stage-optimization-strategy) can be found in `trains/mainNets/ROHYDR.py`.
- The training settings used in our experiments are available in `config/config.json`.

### Upcoming Release (After Paper Acceptance)
We plan to release the following in the next version after the paper is accepted:
- Detailed implementations of all modules, including unimodal representation recovery and multimodal fusion recovery.
- Pretrained weights for quick reproduction of the main results.
- Implementations of all model variants used in the ablation studies on both datasets.
- Comprehensive code comments and documentation to facilitate understanding and extension.
