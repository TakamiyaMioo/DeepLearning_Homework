# 实验 1：Fashion-MNIST 分类与反向传播

本目录包含 Fashion-MNIST 十分类训练代码、NumPy 手写反向传播与梯度检查、配置、结果和 LaTeX 报告。实验比较线性 Softmax、无激活的两层线性网络和 ReLU MLP。此实验不包含 CNN、像素置换或扩散模型；这些内容在 `../Experiment 2/`。

## 目录说明

| 路径 | 内容 |
| --- | --- |
| `src/` | 数据加载、模型、训练循环、评价、随机性控制和绘图 |
| `numpy_bp/` | NumPy MLP 前向/反向传播和中心差分梯度检查 |
| `scripts/` | 训练、模型比较、诊断及总流程入口 |
| `configs/base.yaml` | 数据划分、模型宽度、优化器和训练默认值 |
| `configs/diagnostic.yaml` | 诊断脚本的候选配置 |
| `results/` | 本地运行产生的梯度检查与训练输出 |
| `ClassReport/` | 报告 LaTeX 源码、插图、归档结果和 PDF |
| `requirements.txt` | Python 依赖名称；不锁定全部依赖版本 |

历史服务器运行记录和报告用图保存在 `ClassReport/work/`；这些是实验证据副本。新运行默认写入项目根目录的 `results/` 和 `checkpoints/`。

## 环境

报告所对应的服务器环境为 Linux、Python 3.8、PyTorch 1.10.0+cu113、torchvision 0.11.1+cu113，GPU 为 NVIDIA GeForce RTX 3080 Ti。CUDA 能力来自匹配服务器驱动的 PyTorch wheel；不需要另外安装 CUDA Toolkit。CPU 也可运行，但速度会慢很多。

项目的 `requirements.txt` 只列包名，没有锁定版本。新环境可先创建 Python 环境，再安装与本机驱动兼容的 PyTorch/torchvision；下面给出本实验记录环境的安装示例：

```bash
conda create -n fashion-exp1 python=3.8 -y
conda activate fashion-exp1
python -m pip install --upgrade pip
python -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install numpy scikit-learn matplotlib pandas PyYAML tqdm Pillow
```

若服务器驱动不支持 cu113，请按驱动支持的 CUDA wheel 调整 PyTorch/torchvision 版本；不要为匹配 wheel 单独在服务器安装 CUDA Toolkit。安装后确认设备：

```bash
nvidia-smi
python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 数据与划分

在本目录运行命令。数据根目录默认为 `data/`，Fashion-MNIST 由 `torchvision.datasets.FashionMNIST` 下载并缓存。官方训练集按 `split_seed: 42` 固定分为 50,000 个训练样本和 10,000 个验证样本；索引缓存于 `data/splits/`。官方测试集的 10,000 个样本只用于选定验证集最佳检查点后的最终评价。图像转为 `[0,1]` 张量，不使用随机增强。第一次运行需网络可访问数据源并传 `--download`；数据已经缓存时去掉该选项即可。

## 配置与随机种子

`configs/base.yaml` 当前默认值如下：

| 配置项 | 默认值 |
| --- | ---: |
| seed / split_seed | 42 / 42 |
| train / validation / test | 50,000 / 10,000 / 10,000 |
| hidden_dim / classes | 128 / 10 |
| epochs / batch_size | 25 / 128 |
| optimizer / learning_rate | Adam / 0.001 |
| weight_decay / num_workers | 0 / 2 |
| device / deterministic | auto / true |

模型选择依验证准确率，平手时保留较早的 epoch；测试集不参与模型选择。`seed_everything` 控制 Python、NumPy、PyTorch 和数据加载器的随机性。CUDA 算子、驱动和软件版本不同仍可能产生少量数值差异。

注意：本报告正式三模型表采用 30 epoch、训练种子 42/123/2026；仓库中的 `base.yaml` 是 25 epoch、单 seed 42 的通用默认配置。要按报告预算运行，可用下面命令从同一份基准配置生成三个 30 epoch 配置；只改训练 seed，split seed 始终保持 42：

```bash
for seed in 42 123 2026; do
  sed -e "s/^seed: 42$/seed: ${seed}/" \
      -e 's/^  epochs: 25$/  epochs: 30/' \
      configs/base.yaml > "/tmp/fashion_seed${seed}.yaml"
done
```

## 运行命令

### 梯度检查

只在固定小批量上执行 NumPy 前向/反向与中心差分，不会用 NumPy 训练完整数据集：

```bash
python -m numpy_bp.run_gradient_check
```

输出写入 `results/gradient_check/gradient_check.json` 和 `.csv`。

### 单模型训练

训练入口支持 `linear`、`two_layer_linear` 和 `relu_mlp`。初次下载数据时添加 `--download`；数据缓存后改用 `--no-download`。

```bash
python -m scripts.train --model linear --config configs/base.yaml --download
python -m scripts.train --model two_layer_linear --config configs/base.yaml --no-download
python -m scripts.train --model relu_mlp --config configs/base.yaml --no-download
```

### 三模型比较

比较脚本每次按指定配置顺序运行三个模型。为了复现报告的三种子结果，准备三个配置文件（除训练 seed 外完全相同，且均使用 30 epochs），分别运行：

```bash
python -m scripts.compare_models --config /tmp/fashion_seed42.yaml --download
python -m scripts.compare_models --config /tmp/fashion_seed123.yaml
python -m scripts.compare_models --config /tmp/fashion_seed2026.yaml
```

`compare_models.py` 接受 `--config` 和可选 `--download`，没有 `--no-download` 参数；后两次不传 `--download` 即使用已缓存数据。每个模型/运行的原始配置、逐 epoch 历史、汇总指标、曲线和 checkpoint 保存到 `results/<baseline-or-diagnostic>/<run_id>/` 与 `checkpoints/<model>/<run_id>.pt`。顶层 `results/comparison/` 是最近一次比较的汇总，不会自动合并不同 seed 的多次运行；三种子统计请从各 run 的 `summary.json` 汇总，或参照归档的 `ClassReport/work/results/comparison/`。

### 诊断运行与解释

报告诊断的假设是 ReLU MLP 在第 20 epoch 后出现验证损失回升；实际对照只改变最大 epoch 数（30 对 20），其余条件、seed 和数据划分不变。诊断结论和逐轮结果已归档于 `ClassReport/work/results/diagnostic/`。当前 `configs/diagnostic.yaml` 与 `scripts/diagnose.py` 是一个学习率对照候选脚本，不是报告采用的 epoch 预算诊断；不要将该候选配置的输出误称为报告中的诊断结果。

## 输出与检查点

- `results/baseline/<run_id>/`：`config.json`、`summary.json`、`history.csv`、`history.png`。
- `results/comparison/`：最近一次三模型比较的 CSV/JSON 摘要。
- `results/diagnostic/`：诊断脚本输出；报告实际诊断原始记录在 `ClassReport/work/results/diagnostic/`。
- `results/gradient_check/`：NumPy 梯度检查 CSV/JSON。
- `checkpoints/<model>/<run_id>.pt`：验证准确率最佳 epoch 的模型和优化器状态、配置、seed。

仓库默认忽略新生成的 `data/`、`results/`、`checkpoints/` 和虚拟环境；已归档的实验记录和 PDF 已纳入版本控制。

## 编译报告

需要 XeLaTeX 与 `Settings.tex` 所用宏包。在 `ClassReport/` 下执行：

```bash
cd ClassReport
make
```

中间 PDF 为 `ClassReport/build/main.pdf`；提交版为 `ClassReport/output/pdf/experiment1_report.pdf`。报告源码引用同级的 `../src/`、`../numpy_bp/`、`../scripts/`、`../configs/`，因此不要单独移动 `ClassReport/`。

## 复现范围说明

本 README 中的命令说明当前仓库所含代码。报告正文的三种子、30 epoch 表格以及 30/20 epoch 诊断对应服务器归档结果；通用 `base.yaml` 默认仍为 25 epoch，诊断候选脚本仍为学习率对照。要重新得到报告中的完全相同汇总，需要依照上文准备相同配置并逐 seed 重跑；随机数种子相同也不保证跨硬件和软件版本逐位一致。
