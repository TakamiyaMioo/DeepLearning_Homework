# Fashion-MNIST 分类与反向传播实验

实验 1 项目：Fashion-MNIST 十分类，线性分类器、两层线性模型、ReLU MLP、NumPy 手写反向传播和梯度检查。没有实现 CNN、像素置换或扩散模型。本地阶段不下载数据，也不做完整训练。

## 阿里云 Linux GPU 部署

PyTorch/torchvision 安装命令要按服务器驱动和 CUDA 环境选择，不要盲目安装固定 CUDA wheel。

```bash
unzip cvhomework.zip
cd cvhomework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 按服务器驱动/CUDA 选择 PyTorch 安装命令
pip install -r requirements.txt
python -c "import torch; print(torch.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m numpy_bp.run_gradient_check
python -m scripts.train --model linear --config configs/base.yaml --download
python -m scripts.train --model two_layer_linear --config configs/base.yaml --no-download
python -m scripts.train --model relu_mlp --config configs/base.yaml --no-download
python -m scripts.compare_models --config configs/base.yaml --no-download
python -m scripts.diagnose --config configs/diagnostic.yaml --no-download
python -m scripts.generate_figures
```

首次下载数据传 `--download`，已缓存时使用 `--no-download`。官方训练集固定划分为 50000/10000，索引缓存 `data/splits/`；官方测试集只用于最终评价。图像保持 `[0,1]`，验证和测试不做随机增强。

## 项目和复现

`src/` 是数据、模型、Trainer、评价和绘图；`numpy_bp/` 对合成固定小批量做 NumPy 梯度检查；`scripts/` 按阶段运行。独立 run_id 保存配置、CSV/JSON、checkpoint 和曲线。模型按验证准确率选取，并列时保留较早 epoch。默认随机种子为 42；GPU 浮点及 CUDA 算子可能影响逐位复现。

诊断 YAML 是候选学习率对照，不表示观察到训练问题；需先看初始日志，结论按真实证据填写。报告提纲位于 `report/report_outline.md`，不含虚构指标。

## 打包

在 Windows PowerShell 项目目录运行 `./package.ps1`，生成含 `cvhomework/` 根层级的 ZIP。数据、checkpoint、日志、虚拟环境和缓存均排除。若首次数据下载失败，检查服务器网络和 torchvision 缓存。CUDA OOM 可调低 batch size，但应记录为单独实验设置。
