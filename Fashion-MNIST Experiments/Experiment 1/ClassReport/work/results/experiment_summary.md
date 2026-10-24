# Fashion-MNIST 实验 1 结果汇总

## 实验设置

- 数据集：Fashion-MNIST，训练集 60,000，测试集 10,000；训练集固定划分为 50,000/10,000，split seed=42。
- 正式对比：3 个模型 × 3 个随机种子（42、123、2026），每个模型每个 seed 训练 30 epochs；统一配置，GPU 训练，最佳验证准确率检查点用于测试集评估。
- 诊断实验：ReLU MLP，seed=42，同配置比较 30 epochs 与 20 epochs；依据验证集结果。
- 梯度数值检查和 GPU smoke test 均已完成。

## 多种子结果（均值 ± 样本标准差）

| 模型 | 参数量 | 测试准确率 | Macro-F1 | 每次训练耗时（秒） |
|---|---:|---:|---:|---:|
| Linear | 7850 | 0.8478 ± 0.0006 | 0.8469 ± 0.0006 | 123.2 ± 2.1 |
| Two-layer linear | 101770 | 0.8445 ± 0.0005 | 0.8444 ± 0.0005 | 124.8 ± 3.7 |
| ReLU MLP | 101770 | 0.8840 ± 0.0012 | 0.8846 ± 0.0018 | 122.5 ± 0.9 |

## 产物

- 全部单次运行指标：`results/comparison/all_seed_results.csv` 和 `.json`。
- 跨 seed 汇总：`results/comparison/seed_statistics.csv` 和 `.json`。
- 模型检查点：`checkpoints/<模型名>/<run_id>.pt`。
- 报告图：`report/figures/`；`fig07_model_comparison.png` 展示三 seed 均值和样本标准差。
- NumPy 梯度检查：`results/gradient_check/`；诊断结果：`results/diagnostic/`。

## 运行环境说明

训练实际使用服务器预装的 Python 3.8、PyTorch 1.10.0+cu113 与 CUDA GPU（RTX 3080 Ti），未安装 CUDA toolkit。用户创建的 `cvhomework` Python 3.10 Conda 环境保留未改动；由于其中尚无 PyTorch，正式运行使用服务器预装且已验证可用的 CUDA PyTorch。
