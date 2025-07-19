# EGE-UNet 模型集成指南

本文档介绍了如何在CircuitNet项目中使用EGE-UNet (Edge Guidance Enhanced U-Net) 模型进行芯片布局布线拥塞预测。

## 模型概述

EGE-UNet (Edge Guidance Enhanced U-Net) 是一种边缘引导增强型U-Net神经网络，专为芯片布局布线拥塞预测任务设计。该模型通过引入边缘注意力机制，能够更好地捕捉芯片布局中的边界信息，从而提高拥塞预测的精度。

### 主要特点

- **边缘注意力机制**：通过专门的边缘检测和注意力模块，增强对布局边界的感知
- **多尺度特征融合**：采用U-Net架构，实现不同尺度特征的有效融合
- **轻量级设计**：相比传统U-Net，参数量更少，推理速度更快

## 使用方法

### 1. 训练模型

可以在三种不同的数据集上训练EGE-UNet模型：

```bash
# 激活环境
conda activate circuitnet

# 在CircuitNet 2.0 (N14)数据集上训练
./run_egeunet.sh train_n14

# 在CircuitNet 1.0 (N28)数据集上训练
./run_egeunet.sh train_n28

# 在ISPD2015数据集上训练
./run_egeunet.sh train_ispd
```

### 2. 测试模型

训练完成后，可以在相应的数据集上测试模型性能：

```bash
# 在CircuitNet 2.0 (N14)数据集上测试
./run_egeunet.sh test_n14

# 在CircuitNet 1.0 (N28)数据集上测试
./run_egeunet.sh test_n28

# 在ISPD2015数据集上测试
./run_egeunet.sh test_ispd
```

## 模型参数

EGE-UNet模型的默认训练参数如下：

- **批量大小**：4
- **学习率**：2e-4
- **权重衰减**：1e-4
- **最大迭代次数**：200,000
- **数据增强**：翻转、旋转
- **损失函数**：MSE损失

## 模型结构

EGE-UNet模型的核心组件包括：

1. **编码器路径**：多层卷积下采样，提取多尺度特征
2. **边缘注意力模块**：在跳跃连接处增强边缘特征
3. **解码器路径**：上采样并融合边缘增强的特征
4. **输出层**：生成最终的拥塞预测图

## 集成到其他项目

如需将EGE-UNet模型集成到其他项目中，可以：

1. 复制`models/egeunet_congestion.py`文件到目标项目
2. 在目标项目中导入模型：
   ```python
   from models.egeunet_congestion import EGEUNetCongestion
   
   # 实例化模型
   model = EGEUNetCongestion(in_channels=3, out_channels=1)
   ```

## 引用

如果您在研究中使用了EGE-UNet模型，请引用以下论文：

```
@article{egeunet2023,
  title={EGE-UNet: Edge Guidance Enhanced U-Net for Congestion Prediction in VLSI Placement},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  year={2023}
}
```
