# CircuitNet: Deep Learning for Circuit Design Congestion Prediction

🚀 **先进的深度学习模型用于电路设计中的拥塞预测**

[![GitHub Stars](https://img.shields.io/github/stars/hayaseyuukaa/CircuitNet-ML)](https://github.com/hayaseyuukaa/CircuitNet-ML)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

> **项目状态**: 🔥 积极开发中 | **最后更新**: 2024年7月

## 📖 项目简介

CircuitNet是一个专注于电路设计领域的深度学习项目，主要解决芯片设计中的拥塞预测问题。项目集成了多种先进的神经网络架构，包括U-Net变体、Transformer等模型。

### 🎯 主要目标
- 提高电路拥塞预测的准确性
- 优化NRMSE和SSIM等关键指标
- 提供统一的训练和评估框架
- 支持多种数据集（N14、N28、ISPD2015）

## 🏗️ 模型架构

### 支持的模型
- **ibUNet**: 改进的U-Net架构，专门优化用于拥塞预测
- **RouteNet**: 基于图神经网络的路由预测模型
- **GPDL**: 全卷积网络用于密度预测
- **GraphSAGE**: 图采样和聚合网络
- **Swin Transformer**: 基于窗口的Transformer架构
- **VM-UNet**: 视觉Mamba U-Net混合架构

### 评估指标
- **NRMSE**: 归一化均方根误差
- **SSIM**: 结构相似性指数
- **PSNR**: 峰值信噪比
- **Peak NRMS**: 峰值归一化均方根误差

## 🚀 快速开始

### 环境要求
```bash
# 创建conda环境
conda env create -f simple_env.yml
conda activate circuitnet

# 或使用pip安装依赖
pip install torch torchvision numpy scipy scikit-image matplotlib
```

### 训练模型
```bash
# 训练ibUNet模型（推荐）
./run_ibunet.sh train_n28

# 训练其他模型
./run_routenet.sh train_n28
./run_gpdl.sh train_n28
```

### 测试模型
```bash
# 测试训练好的模型
./run_ibunet.sh test_n28
```

## 📊 数据集

### 支持的数据集
- **N14**: 14nm工艺节点数据集
- **N28**: 28nm工艺节点数据集
- **ISPD2015**: ISPD竞赛标准数据集

### 数据格式
- 输入：电路布局特征图
- 输出：拥塞密度预测图
- 格式：NumPy数组或PyTorch张量

## 🔧 项目结构

```
CircuitNet6/
├── CircuitNet-main/           # 主要代码目录
│   ├── run_*.sh              # 各模型训练脚本
│   ├── configs.py            # 配置文件
│   ├── train.py              # 统一训练脚本
│   ├── test.py               # 统一测试脚本
│   └── utils/                # 工具函数
├── work_dir/                 # 训练输出目录
├── git_workflow.sh           # Git工作流脚本
└── README.md                 # 项目说明
```

