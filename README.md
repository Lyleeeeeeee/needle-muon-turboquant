# Needle: Muon + TurboQuant project

这是 10-714 project 的起始框架。已经合并完成的 HW1 自动求导、HW2 网络层／优化器／数据处理，以及 HW3 NDArray 和 CPU／CUDA 后端源码。

当前阶段：框架整合和训练验证完成。Muon 与 TurboQuant 是接下来的两条并列实施路线，具体任务见 [实施计划](docs/implementation_plan.md)。

## 安装和检查

在仓库根目录运行：

```bash
python -m pip install -r requirements-dev.txt
python tools/build_cpu.py
python -m pytest -q
```

`build_cpu.py` 为当前 Python 编译 HW3 原生 CPU 扩展，支持 macOS 和 Linux。有 pybind11 时使用其头文件，也可复用已安装 PyTorch 中的 pybind11 头文件；训练本身由 Needle 完成。默认使用 `nd` 后端和 `cpu_numpy()` 设备，无需编译即可运行基础计算；`--device cpu` 使用 C++ 后端。

本机通过了 18 项集成检查，覆盖 NumPy 和 HW3 NDArray 后端的梯度、稳定损失函数、归一化、优化器、原生 CPU 训练及训练命令。这是 project 的集成检查，并非完整的作业评分测试。CI／其他 Python 环境尚待验证。CUDA 源码保留，尚未在 NVIDIA GPU 上验证。

## 跑通训练

准备 MNIST 的四个原始 `.gz` 文件，放在同一目录，然后运行：

```bash
python apps/train_baseline.py --data-dir /path/to/mnist --device cpu --train-limit 1024 --eval-limit 256 --epochs 3 --output results/baseline.json
```

在当前课程工作区中，脚本默认使用相邻 `hw0/data`，可以省略 `--data-dir`。`--device numpy` 则使用 HW3 的 NumPy 存储设备。模型是 `784 → 128 → 64 → 10` 的小型 MLP；`model.features(x)` 提供用于后续检索的 64 维特征。当前使用 AdamW。

这次 MNIST 小规模训练只用于检查框架，结果保存在 [基线运行记录](reports/baseline_mnist_cpu.json)。正式 project 将使用 Fashion-MNIST，固定数据划分、训练预算和随机种子后，再比较算法性能。

## 来源与整合

原始归档保存在本地 `sources/`，文件来源和 SHA-256 在 [来源记录](sources/manifest.json) 中。HW2 归档已经包含完成的 HW1 自动求导和算子；整合时保留这些实现，并接入 HW3 后端。适配包括设备选择、初始化、转置／归约、稳定 log-sum-exp，以及 ReLU 梯度的设备一致性。

保留原 HW2 `Adam` 的行为，另外增加了具有独立权重衰减的 `AdamW`，作为 Muon 的比较基础。原作业目录与原归档独立保存。

仓库包含课程框架与作业实现，按课程要求保持私有。
