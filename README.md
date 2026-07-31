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
**Wai Keung Wong**<sup>1†</sup>,
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
