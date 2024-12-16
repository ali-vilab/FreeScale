<p align="center">
    <img src="assets/icon.png" width="200">
</p>

## ___***FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion***___

### 🔥🔥🔥 FreeScale is a tuning-free method for higher-resolution visual generation, unlocking the 8k image generation!

<div align="center">
 <a href='https://arxiv.org/abs/2412.09626'><img src='https://img.shields.io/badge/arXiv-2412.09626-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='http://haonanqiu.com/projects/FreeScale.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a>


_**[Haonan Qiu](http://haonanqiu.com/), [Shiwei Zhang*](https://scholar.google.com/citations?user=ZO3OQ-8AAAAJ), [Yujie Wei](https://weilllllls.github.io/), [Ruihang Chu](https://ruihangchu.com/), [Hangjie Yuan](https://jacobyuan7.github.io/), 
<br>
[Xiang Wang](https://scholar.google.com/citations?user=cQbXvkcAAAAJ), [Yingya Zhang](https://scholar.google.com/citations?user=16RDSEUAAAAJ), and [Ziwei Liu†](https://liuziwei7.github.io/)**_
<br><br>
(* Project Leader, † Corresponding Author)

From Alibaba Group and Nanyang Technological University.

<img src="assets/fig_teaser.png">
</div>

## ⚙️ Setup

### Install Environment via Anaconda (Recommended)
```bash
conda create -n freescale python=3.8
conda activate freescale
pip install -r requirements.txt
```


## 💫 Inference 
### 1. Higher-Resolution Text-to-Image

1) Download the pre-trained SDXL checkpoints from [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
2) Modify the `run_freescale.py` and input the following commands in the terminal.
```bash
  python run_freescale.py

  # resolutions_list: resolutions for each stage of self-cascade upscaling.
  # cosine_scale: detail scale, usually 1.0 ~ 2.0. For 8k image generation, cosine_scale <= 1.0 is recommended.
```
<img src="assets/fig_diff8k.png">

### 2. Flexible Control for Detail Level

1) Download the pre-trained SDXL checkpoints.
2) Modify the `run_sdxl.py` and generate the base image with the original resolutions.
```bash
  python run_sdxl.py
```
3) Put the generated image into folder `imgen_intermediates`.
4) (Optional) Generate the mask using other segmentation models (e.g., [Segment Anything](https://huggingface.co/spaces/Xenova/segment-anything-web)) and put the mask into folder `imgen_intermediates`.
5) Modify the `run_freescale_imgen.py` and generate the final image with the higher resolutions.
```bash
  python run_freescale_imgen.py

  # resolutions_list: resolutions for each stage of self-cascade upscaling.
  # cosine_scale: detail scale for foreground, usually 2.0 ~ 3.0. 
  # cosine_scale_bg: detail scale for background, usually 0.5 ~ 1.0.
```
<img src="assets/fig_mask.png">

### 3. Tips:
1. Generating 8k (8192 x 8192) images will cost around 56 GB and 1 hour on NVIDIA A800. 
2. Set `fast_mode = True` can significantly shorten the time but lead to some loss of quality especially for 8k image generation.
3. For 8k image generation, `cosine_scale <= 1.0` is recommended. Or use the Flexible Control for Detail Level function and set a small `cosine_scale_bg` (e.g. 0.5) for areas with artifacts. 