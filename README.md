# NTIRE 3D Reconstruction Dehazing: Integrated Multi-Stage Pipeline

本项目是针对 3D 重建去雾比赛开发的自动化处理管线。系统结合了物理先验去雾、3D 高斯泼溅 、去雾模型增强技术。

## 🌟 系统架构图
1. **预处理 (IHDCP)**: 基于物理先验去雾，去除首层浓雾。
2. **3D 重建 (3D-UIR)**: 利用 3D Gaussian Splatting 进行新视角合成，产出多视角 hazy 图像及深度图。
3. **2D 增强**: 使用多种去雾模型对结果进行增强。
  
---

## 🛠️ 环境配置

本项目需要两个 Conda 环境以兼容不同阶段的代码依赖。

### 1. 渲染环境 (gaussian_splatting)
用于运行 3D-UIR 渲染逻辑。
- 主要参考: 3D Gaussian for Underwater 3D Scene Reconstruction via Physics-Based Appearance-Medium Decoupling
- Python 3.10 / PyTorch 2.1.0 / CUDA 11.8
- 核心依赖: `diff-gaussian-rasterization`, `simple-knn`

### 2. 去雾环境 (ipc)
用于运行所有深度学习去雾模型及扩散模型。
- 主要参考: Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing
- Python 3.8 / PyTorch 1.13+ / CUDA 11.7+
- 核心依赖: `diffusers`, `transformers`, `opencv-python`, `pytorch-msssim`

---

## 📦 权重准备

由于模型权重体积较大，请从以下链接下载并放置于对应目录：
- **下载链接**: [https://pan.baidu.com/s/1bFYhcMgtUdhyVWAX5E1Q6g?pwd=ib4g 提取码: ib4g]
- **放置路径**:
  - `weights/` (集成权重包)
  - `checkpoints/difix_hf/` (扩散模型目录)
---

## 🚀 运行指南

整个管线已实现高度自动化。只需在 **渲染环境** 下启动渲染，系统会自动触发后续所有去雾步骤。

### 第一步：Matlab 预处理
使用 `matlab 运行 IHDCP` 对原始图像进行首轮处理。

### 第二步：3D 渲染与自动去雾
将上一步的结果输入3D-UIR进行训练，训练好的点云文件位于 3D-UIR/output/<场景名> 目录下。
```bash
cd 3D-UIR
# 运行特定场景的渲染，以 tsubaki 场景为例，系统会自动寻找 output/tsubaki 文件夹（渲染完成后会触发去雾）
python render.py -m output/tsubaki
```

### 第三步：获取最终结果
所有处理完成后，请前往以下路径提取成品图像：
results/<场景名>/final_submission/
