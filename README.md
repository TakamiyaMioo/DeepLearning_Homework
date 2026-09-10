# MNIST DDPM

一个小型、可上传服务器运行的 PyTorch DDPM 实验工程。实现内容对应实验要求：

- 线性 beta 噪声日程，默认 1000 个扩散步；
- `q(x_t|x_0)` 一步加噪采样；
- 参数量不超过 5M 的时间条件 Tiny U-Net；
- 以 MSE 预测噪声 `epsilon_theta`；
- ancestral sampling，从纯高斯噪声逐步生成 8×8 样本网格；
- 使用同一训练好的 1000 步日程，对 1000/200/50 reverse steps 做均匀跳步对比。

本项目训练环境为 AutoDL 单张 NVIDIA GeForce RTX 3080 Ti，报告指导教师为王兴刚。

## 环境

```bash
conda create -n mnist-ddpm python=3.10 -y
conda activate mnist-ddpm
pip install -r requirements.txt
```

首次运行训练脚本时会自动下载 MNIST。建议服务器使用 CUDA，并根据显存调整 `--batch-size`。

## 训练（仅上传后执行）

```bash
python train.py --epochs 20 --batch-size 128 --timesteps 1000 \
  --data-dir data --output-dir outputs/train
```

脚本会保存 checkpoint、`loss.png` 和最终样本图。训练前只需检查参数，不要在本地运行上述命令即可。

## 采样与步数对比

```bash
python sample.py --checkpoint outputs/train/epoch_020.pt \
  --output outputs/sample.png --num-samples 64

python compare_steps.py --checkpoint outputs/train/epoch_020.pt \
  --output-dir outputs/compare --num-samples 64
```

对比结果为 `steps_1000.png`、`steps_200.png`、`steps_50.png`。这里的 200/50 步是从训练使用的 1000 步噪声日程中均匀抽取 reverse steps，便于比较采样质量与耗时。

## 已完成实验与报告

基于 `outputs/train/epoch_020.pt`，分析脚本可重新生成报告图表和采样耗时：

```bash
python analysis_experiment.py --checkpoint outputs/train/epoch_020.pt \\
  --output-dir ClassReport/figures --num-samples 64 --device cuda
```

报告源码位于 `ClassReport/main.tex`，使用 XeLaTeX 编译：

```bash
cd ClassReport
make
```

生成的 PDF 位于 `ClassReport/build/main.pdf`。报告包含损失曲线、8×8 生成样本网格、1000/200/50 步样本对比和耗时分析。

## 目录

```text
MNIST_DDPM/
├── train.py            # 训练入口
├── sample.py           # 1000 步 ancestral sampling
├── compare_steps.py    # 1000/200/50 步对比
├── analysis_experiment.py # 生成报告图表和采样耗时
├── ClassReport/         # LaTeX 实验报告
├── requirements.txt
└── src/
    ├── diffusion.py    # 前向过程与反向采样
    ├── model.py        # Tiny U-Net
    └── utils.py
```

## 说明

当前版本没有自动执行训练，也没有预下载数据或生成 checkpoint；这些文件会在服务器运行时产生。
