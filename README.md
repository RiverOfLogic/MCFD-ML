# MCFD-ML

基于多约束特征解耦和元学习的旋转机械跨域故障诊断 (Multi-Constraint Feature Disentanglement with Meta-Learning for Cross-Domain Machinery Fault Diagnosis)

## 📋 项目简介

本项目实现了一种基于深度学习的多源域泛化方法，用于旋转机械故障诊断。通过结合多种域自适应技术，提高模型在未见目标域上的泛化能力。

## 🔧 主要方法

项目实现了多种域自适应/域泛化算法：

- **MCFD-ML**: 主要方法，集成多种域适应策略
- **DANN** (Domain Adversarial Neural Network): 域对抗神经网络
- **CDAN** (Conditional Domain Adversarial Network): 条件域对抗网络
- **MCD** (Maximum Classifier Discrepancy): 最大分类器差异
- **MLDG** (Meta-Learning Domain Generalization): 元学习域泛化方法
- **CDDG** (Causal Disentanglement Domain Generalization): 因果解耦领域泛化
- **ERM** (Empirical Risk Minimization): 经验风险最小化（基线方法）

## 📁 项目结构

```
MEDG_DA/
├── src/                    # 源代码目录
│   ├── MEDG.py            # 主训练脚本
│   ├── MEDGNet.py         # 网络模型定义
│   ├── DANN.py            # DANN 实现
│   ├── CDAN.py            # CDAN 实现
│   ├── MCD.py             # MCD 实现
│   ├── ERM.py             # ERM 实现
│   ├── MyNewDataset.py    # 数据集加载器
│   ├── config.py          # 配置文件
│   └── ...
├── models/                 # 预训练模型
│   ├── task2_43_99.35.pt
│   ├── task4_42_99.76.pt
│   └── task7_43_98.92.pt
├── logs/                   # 训练日志
│   ├── MEDG_training.log
│   ├── DANN_training.log
│   └── ...
└── README.md              # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- matplotlib

### 配置参数

在 `src/config.py` 中配置以下参数：

```python
dataset = "MAFAULDA"  # 或 "DIRG"
TASK = 7              # 任务编号 (1-8)

# MEDG 超参数
num_classes = 7
epochs = 100
batch_size = 128
lr = 0.0005
weight_outer = 0.5
weight_coral = 0.3
weight_adv = 1
weight_domainacc = 0.2
weight_HSIC = 0.1
weight_rec = 0.2
```

### 训练模型

```bash
cd src
python MEDG.py --seed 42
```

### 任务划分

项目支持 8 种不同的任务划分，每种任务定义了源域和目标域的组合：

- **Task 1-4**: DIRG 数据集的不同转速组合
- **Task 5-8**: MAFAULDA 数据集的不同工况组合

## 📊 实验结果

部分任务的测试准确率：

| 任务 | Seed | 准确率 | Macro F1 | Weighted F1 |
|------|------|--------|----------|-------------|
| Task 2 | 43 | 99.35% | 0.9935 | 0.9935 |
| Task 4 | 42 | 99.76% | 0.9975 | 0.9976 |
| Task 7 | 43 | 98.92% | 0.9892 | 0.9892 |
| Task 5 | 42/43 | 100.00% | 1.0000 | 1.0000 |

详细结果请查看 `logs/` 目录下的训练日志。

## 📈 可视化

项目支持以下可视化输出：

- **t-SNE 特征可视化**: `tsne_output.pdf`
- **测试结果图表**: `test_results*.pdf`

## 🔑 核心特性

1. **多尺度特征提取**: 使用多尺度残差瓶颈块模拟 Transformer 的多频提取能力
2. **坐标注意力机制**: 增强模型对关键特征的捕捉能力
3. **元学习策略**: 通过内外循环优化提升泛化性能
4. **多损失融合**: 结合 CORAL、对抗损失、HSIC 等多种损失函数
5. **梯度反转层**: 实现域不变特征学习

## 📝 数据集

支持的数据集：
- **DIRG**: 不同转速下的轴承故障数据
- **MAFAULDA**: 多工况机械故障数据集

## 📄 许可证

本项目仅供学术研究使用。

## 📧 联系方式

如有问题，请提交 Issue 或联系作者。