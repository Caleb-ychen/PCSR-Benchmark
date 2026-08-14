# PCSR-Benchmark
Code release for Beyond Localization: A Comprehensive Benchmark of Perspective-Conditioned Spatial Reasoning in MLLMs from Omnidirectional Images [ACMMM 2026]

<div align="center">

# Beyond Localization: A Comprehensive Benchmark of Perspective-Conditioned Spatial Reasoning in MLLMs from Omnidirectional Images
## A Comprehensive Benchmark of Perspective-Conditioned Spatial Reasoning

<img src="docs/resources/fig1.svg" width="90%"/>

<br>

[![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/xxxx.xxxxx)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://your-project-page.github.io/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/your-dataset)
[![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/yourname/yourrepo)

**Yuangong Chen**<sup>1</sup>,
**Waikeung Wong**<sup>1†</sup>,
**Jiaxing Li**<sup>2</sup>,
**Ioannis Patras**<sup>3</sup>,
**Xu Zheng**<sup>4</sup>

<sup>1</sup>The Hong Kong Polytechnic University, Hong Kong SAR, China &emsp;
<sup>2</sup>Guangzhou University, Guangzhou, China &emsp;
<sup>3</sup>Queen Mary, University of London, London, United Kingdom &emsp;
<sup>4</sup>Great Bay University, Dongguan, China

<br>
<sup>†</sup>Corresponding author

</div>



## News
- `2026-07-30` Release evaluation code and benchmark.
- `2026-07-10` Paper accepted by ACM MM 2026.

## 📖 Abstract

Understanding spatial relationships from omnidirectional (360°) images is a fundamental challenge for Multimodal Large Language Models (MLLMs), yet existing benchmarks largely focus on object localization while overlooking the **perspective-conditioned** nature of spatial reasoning. In this work, we introduce **PCSR-Benchmark**, a comprehensive benchmark designed to evaluate MLLMs on perspective-conditioned spatial reasoning tasks from omnidirectional images. Our benchmark covers *(1)* egocentric and allocentric reasoning, *(2)* multi-view consistency, and *(3)* fine-grained relational understanding across diverse indoor and outdoor scenes. We evaluate a wide range of state-of-the-art open-source and proprietary MLLMs, revealing that current models exhibit substantial limitations when reasoning beyond simple localization. We further provide detailed analyses on failure modes and propose insights for developing next-generation spatially-aware MLLMs. We hope PCSR-Benchmark will serve as a rigorous testbed to advance research on spatial reasoning in omnidirectional visual understanding.

<!-- Optional: teaser figure under abstract -->
<p align="center">
  <img src="docs/resources/abstract_teaser.svg" width="85%"/>
</p>



## Contents
- [Overview](#overview)
- [Benchmark](#benchmark)
- [Method](#method)
- [Results](#results)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Overview
**Motivation.** Existing work focuses on xxx, but ignores xxx.

**Our contribution.**
- We introduce ...
- We benchmark ...
- We show ...

<img src="assets/overview.png" width="100%"/>

## Benchmark
 we introduce PCSR-Bench, a benchmark designed to test whether models can recompute spatial relations under changed observer conditions.

<img src="docs/resources/benchmark_generation_pipeline-v4.svg" width="100%"/>

## Results
We evaluate on xxx under xxx setting.

<img src="docs/resources/table1.png" width="70%"/>

## Evaluation
```python
from datasets import load_dataset
dataset = load_dataset("your-name/your-dataset")
print(dataset)
