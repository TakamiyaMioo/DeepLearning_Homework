# 实验 2：Fashion-MNIST 卷积分类、结构先验与扩散生成

本目录收录实验二报告、报告图表、分类运行记录和扩散采样评价数据。实验包含原图/固定像素置换图上的 MLP 与 CNN 对照，以及无条件 DDPM 的训练与采样评价。

> **复现状态：**当前目录包含报告和实验结果，但没有实验二的训练脚本、去噪器源码、模型 checkpoint 或完整依赖锁定文件。因此现在可以重新编译报告、检查记录和图表，但不能从本目录重新训练分类模型/扩散模型，也不能从 checkpoint 重做采样。这里不会提供臆造的 `train.py` 命令。需要重新训练时，必须先从原训练服务器补入对应源码、checkpoint 或可访问的下载位置，并将其版本信息记录在此 README。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `main.tex`、`Settings.tex`、`Makefile` | 报告源文件和 XeLaTeX 构建入口 |
| `images/exp2/` | 分类曲线、准确率比较、混淆矩阵、错误样本、扩散过程和样本评价图 |
| `work/server_artifacts/results/exp2/` | 逐种子 CSV/JSON、训练历史、日志、架构表与汇总 |
| `work/server_artifacts/results/exp2/diffusion/metadata.json` | 扩散训练/采样配置、随机种子和耗时记录 |
| `work/server_artifacts/results/exp2/diffusion/fixed_samples.pt` | 固定初始噪声与采样批次记录 |
| `work/server_artifacts/results/exp2/diffusion/evaluation.pt` | 固定采样批次的评价数据 |
| `output/pdf/experiment2_report.pdf` | 提交版 PDF |

`.pt` 文件是结果评价/固定噪声数据，不是训练好的模型 checkpoint。

## 环境记录

训练记录确认分类和扩散任务运行在 CUDA GPU 上；扩散训练使用 50,000 个样本、基础通道数 32 和 100 个扩散步。现有日志与 `metadata.json` 没有保留确切的 Python、PyTorch、torchvision 版本、GPU 型号或驱动版本。因此无法保证按某个版本号新建环境就逐数复现；请不要把报告中的 CUDA 设备记录误认为一份完整的软件环境锁。

重新编译报告需要：

- XeLaTeX（例如 TeX Live）；
- `Settings.tex` 导入的 LaTeX 宏包，以及 `pgfplots`、`tikz`；
- 报告引用的图片和训练 CSV（均已包含在本目录）。

仅编译报告不需要 PyTorch、Fashion-MNIST 原始数据或 GPU。重新运行训练则必须先补齐本目录目前缺失的源码、checkpoint/权重说明及实际 Python 依赖版本。

## 报告采用的分类实验配置

| 项目 | 配置 |
| --- | --- |
| 数据集 | Fashion-MNIST，灰度 $28\times28$，10 类 |
| 划分 | 沿用实验一固定划分：50,000 train / 10,000 validation；官方 10,000 test 仅最终评价 |
| 输入 | 原图，或统一固定像素置换图；置换同时应用于 train/validation/test |
| 置换种子 | 20261024；同一 784 像素排列用于所有数据子集 |
| 模型 | MLP：784→128 ReLU→10；CNN：Conv(1,32)→ReLU→MaxPool→Conv(32,64)→ReLU→MaxPool→FC(3136,128)→ReLU→FC(128,10) |
| 训练预算 | 每个模型/输入条件 30 epochs，batch size 128 |
| 优化器 | Adam，learning rate 0.001，weight decay 0 |
| 训练种子 | 42、123、2026；报告给出三种子均值和样本标准差 |
| 检查点选择 | 按验证准确率选最佳 epoch；测试集不用于选模 |

单次模型参数量：MLP 101,770；CNN 421,642。原图和置换图的详细逐种子结果在 `comparison_all_seeds.csv`，均值/标准差在 `comparison_statistics.csv`。每个 `original_*` 或 `permuted_*` 的 `*_history.csv` 记录逐 epoch 曲线；`*_confusion.csv`、`*_per_class.csv` 和 `*_errors.csv` 是对应 seed 的测试指标。训练过程日志为 `classification_training.log` 及按种子拆分的日志文件。

## 扩散模型训练与采样记录

该模型是无类别条件的简化 DDPM。训练图像经过前向加噪后，时间步条件去噪器预测噪声，以噪声均方误差训练；采样从固定纯高斯噪声开始，依 100 个反向扩散步逐步更新。

| 项目 | 记录 |
| --- | --- |
| 训练样本数 | 50,000（固定训练子集） |
| 去噪器基础通道 / 参数量 | 32 / 148,673 |
| 扩散步数 / epoch | 100 / 30 |
| 设备 | CUDA GPU（具体型号未保存在元数据中） |
| train seed | 42 |
| sampling seed | 20261024 |
| trajectory noise seed | 314159 |
| 固定采样批次 | 100 张，初始噪声保存在 `diffusion/fixed_samples.pt` |
| 训练耗时 | 129.58 秒 |
| 噪声预测 MSE | epoch 1：0.4021；epoch 30：0.1402 |
| 完整采样耗时 | 两份记录为 0.158 秒和 0.205 秒/100 张（约 1.58–2.05 ms/张） |

噪声 MSE 是训练目标记录，不是生成质量指标；本实验没有单独的扩散验证集指标。CNN 代理分类器对固定 100 张样本预测为 Bag 86、Shirt 5、T-shirt/top 3、Trouser 3、Sandal 2、Pullover 1，覆盖 6/10 类，表明有严重的类别集中。代理标签不是人工真值；报告也包含随机样本网格、前向加噪/反向去噪过程、训练集最近邻及最低置信度样本。

## 可执行命令

### 重新编译 PDF

从本目录运行：

```bash
make
```

Makefile 连续运行两次 XeLaTeX，生成 `build/main.pdf` 和辅助文件。版本控制中的提交版 PDF 是 `output/pdf/experiment2_report.pdf`。在没有 `make` 的环境也可手动运行：

```bash
mkdir -p build
xelatex -interaction=nonstopmode -output-directory=build main.tex
xelatex -interaction=nonstopmode -output-directory=build main.tex
```

该操作仅编译报告，不启动训练。

### 浏览 CSV 记录

安装 pandas 后可查看三种子分类汇总：

```bash
python -c "import pandas as pd; p='work/server_artifacts/results/exp2/comparison_statistics.csv'; print(pd.read_csv(p).to_string(index=False))"
python -c "import pandas as pd; p='work/server_artifacts/results/exp2/comparison_all_seeds.csv'; print(pd.read_csv(p).to_string(index=False))"
```

训练与采样参数可直接读取 JSON：

```bash
python -c "import json; print(json.load(open('work/server_artifacts/results/exp2/diffusion/metadata.json', encoding='utf-8')))"
```

### 训练与采样

当前快照不包含实验二的 Python 入口脚本、模型定义或训练 checkpoint，故没有可执行的训练/采样命令。请勿把上面的 `make` 当成训练命令，也不要仅凭报告配置自行拼造参数。补齐原始服务器项目后，应将实际入口、完整命令行、版本锁定和 checkpoint 路径写在本节，并验证其产生的日志与当前 `results/exp2/` 一致。

## 文件完整性与复现边界

- 图表是报告使用的固定结果图；原始逐 epoch 和分类指标 CSV/JSON 保留在 `work/server_artifacts/results/exp2/`。
- 最近邻从固定训练子集每隔 5 张抽取的 10,000 张图中搜索，并使用原始像素 $L_2$ 距离；不能排除子集之外的语义近邻。
- 生成类别由独立 CNN 代理预测，代理可能对模糊/异常样本过度自信；类别覆盖不是人工标注质量。
- 若重跑了训练，请记录 commit、配置文件、依赖版本、GPU 型号、驱动版本、全部随机种子和完整命令，且不要覆盖已有服务器结果记录。
