# 自适应GMM置信度策略使用指南

## 概述

自适应GMM（Gaussian Mixture Model）策略是对原始GMM置信度过滤的改进，主要解决了原始方法中阈值突变导致的训练不稳定问题。

### 核心改进

1. **时间平滑**：使用指数移动平均（EMA）平滑阈值变化
2. **自适应频率**：根据训练阶段动态调整更新频率
3. **渐进式预热**：早期阶段逐步增加更新间隔

## 使用方法

### 1. 基础使用

```bash
python train_uni.py \
    -s /path/to/dataset \
    -m /path/to/output \
    --use_adaptive_gmm \
    --iterations 10000
```

### 2. 高级参数配置

```bash
python train_uni.py \
    -s /path/to/dataset \
    -m /path/to/output \
    --use_adaptive_gmm \
    --gmm_update_freq 100 \      # GMM更新基础频率（默认100）
    --gmm_momentum 0.95 \        # EMA动量系数（默认0.95）
    --gmm_warmup_iters 500 \     # 预热迭代次数（默认500）
    --gmm_adaptive_freq \        # 启用自适应频率调整
    --iterations 10000
```

### 3. 参数说明

- **--use_adaptive_gmm**: 启用自适应GMM策略
- **--gmm_update_freq**: 基础更新频率，建议保持100（根据您的反馈）
- **--gmm_momentum**: 动量系数，范围[0,1]，越大越平滑
  - 0.95：保留95%历史值（推荐）
  - 0.9：更快响应变化
  - 0.99：更稳定但响应慢
- **--gmm_warmup_iters**: 预热期，在此期间逐渐增加更新间隔
- **--gmm_adaptive_freq**: 根据阈值稳定性自动调整更新频率

## 对比实验

### A/B测试脚本

```bash
# 使用提供的脚本
bash run_adaptive_gmm.sh

# 或手动运行对比实验
# 实验A：自适应GMM
python train_uni.py -s data -m output/adaptive --use_adaptive_gmm

# 实验B：原始GMM（基线）
python train_uni.py -s data -m output/baseline
```

## 可视化分析

使用提供的可视化工具分析GMM行为：

```python
from utils.gmm_visualization import visualize_gmm_thresholds
from utils.adaptive_gmm import AdaptiveGMMPolicy

# 训练后分析
gmm = AdaptiveGMMPolicy()
stats = gmm.get_stats()

# 可视化阈值变化
visualize_gmm_thresholds(
    threshold_history=stats['threshold_history'],
    variance_history=stats['variance_history'],
    save_path="output/gmm_analysis.png"
)
```

## 预期效果

### 1. 训练稳定性提升
- 损失曲线更平滑
- 减少训练震荡
- 收敛更快

### 2. 深度质量改善
- 置信度过滤更合理
- 深度估计误差降低
- 边缘区域处理更好

### 3. 计算效率
- 保持原有的计算效率（GMM拟合频率不变）
- 自适应频率可在后期减少计算

## 常见问题

### Q1: 为什么保持100次迭代的更新频率？
A: 根据您的反馈，更频繁的更新（如每次迭代）会导致：
- 计算开销增加
- 阈值不稳定
- 训练效果下降

### Q2: momentum参数如何选择？
A: 建议值：
- 一般场景：0.95
- 数据噪声大：0.97-0.99（更稳定）
- 快速实验：0.9（更快适应）

### Q3: 如何判断改进是否有效？
A: 观察以下指标：
- TensorBoard中的`train/gmm_variance`曲线（越低越好）
- 训练损失的平滑度
- 最终的PSNR/SSIM指标

## 技术细节

### 算法流程

```python
if iteration % update_freq == 0:
    # 1. 计算新的GMM阈值
    new_threshold = gmm_policy(confidence_scores)
    
    # 2. 应用EMA平滑
    if running_threshold is None:
        running_threshold = new_threshold
    else:
        running_threshold = momentum * running_threshold + (1-momentum) * new_threshold
    
    # 3. 使用平滑后的阈值
    mask = confidence_scores > running_threshold
```

### 自适应频率逻辑

- **预热期**（0-500迭代）：从50逐渐增加到100
- **稳定期**（500+迭代）：
  - 阈值稳定时：增加到150
  - 阈值波动时：减少到70
  - 正常情况：保持100

## 引用

如果这个方法对您的研究有帮助，请考虑引用：

```
基于时间自适应GMM的稀疏视角3D高斯映射深度置信度过滤
``` 