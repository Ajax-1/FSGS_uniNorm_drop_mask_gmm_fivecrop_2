#!/usr/bin/env python3
"""
GMM策略调试工具
用于分析自适应GMM的实际行为和潜在问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.adaptive_gmm import AdaptiveGMMPolicy, gmm_policy as original_gmm_policy
import os

def analyze_gmm_behavior(confidence_values, iterations=1000):
    """
    分析GMM策略在给定置信度数据上的行为
    
    Args:
        confidence_values: 置信度值数组
        iterations: 模拟的迭代次数
    """
    print("=== GMM Behavior Analysis ===")
    
    # 创建自适应GMM实例
    adaptive_gmm = AdaptiveGMMPolicy(
        base_update_freq=100,
        momentum=0.95,
        warmup_iters=500,
        adaptive_freq=True
    )
    
    # 存储结果
    adaptive_thresholds = []
    original_thresholds = []
    update_frequencies = []
    
    for iteration in range(0, iterations, 100):  # 每100次迭代测试一次
        # 模拟置信度数据
        conf_tensor = torch.tensor(confidence_values, dtype=torch.float32)
        
        # 自适应GMM
        adaptive_threshold, updated = adaptive_gmm.get_threshold(conf_tensor, iteration)
        adaptive_thresholds.append(adaptive_threshold)
        
        # 原始GMM
        if iteration % 100 == 0:
            original_threshold = original_gmm_policy(conf_tensor)
            original_thresholds.append(original_threshold)
        else:
            original_thresholds.append(original_thresholds[-1] if original_thresholds else 0.5)
        
        # 记录更新频率
        update_freq = adaptive_gmm.get_adaptive_update_freq(iteration)
        update_frequencies.append(update_freq)
        
        if iteration % 200 == 0:
            print(f"Iter {iteration:4d}: Adaptive={adaptive_threshold:.4f}, "
                  f"Original={original_threshold:.4f}, UpdateFreq={update_freq}")
    
    return adaptive_thresholds, original_thresholds, update_frequencies

def visualize_debug_results(adaptive_thresholds, original_thresholds, update_frequencies, save_path="debug_gmm"):
    """可视化调试结果"""
    os.makedirs(save_path, exist_ok=True)
    
    iterations = np.arange(len(adaptive_thresholds)) * 100
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 阈值对比
    axes[0].plot(iterations, adaptive_thresholds, 'b-', label='Adaptive GMM', linewidth=2)
    axes[0].plot(iterations, original_thresholds, 'r-', label='Original GMM', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Threshold')
    axes[0].set_title('Threshold Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 阈值差异
    threshold_diff = np.array(adaptive_thresholds) - np.array(original_thresholds)
    axes[1].plot(iterations, threshold_diff, 'g-', label='Adaptive - Original', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Threshold Difference')
    axes[1].set_title('Adaptive vs Original Difference')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 更新频率
    axes[2].plot(iterations, update_frequencies, 'm-', label='Update Frequency', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Update Frequency')
    axes[2].set_title('Adaptive Update Frequency')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gmm_debug_analysis.png'), dpi=150)
    plt.close()
    
    print(f"Debug visualization saved to {save_path}/gmm_debug_analysis.png")

def test_different_parameters():
    """测试不同参数组合的效果"""
    print("\n=== Testing Different Parameters ===")
    
    # 生成模拟置信度数据
    np.random.seed(42)
    confidence_values = np.random.beta(2, 2, 10000)  # 模拟置信度分布
    
    parameter_configs = [
        {"momentum": 0.95, "name": "High Momentum (0.95)"},
        {"momentum": 0.9, "name": "Medium Momentum (0.9)"},
        {"momentum": 0.8, "name": "Low Momentum (0.8)"},
        {"momentum": 0.7, "name": "Very Low Momentum (0.7)"},
    ]
    
    results = {}
    
    for config in parameter_configs:
        print(f"\nTesting {config['name']}...")
        
        adaptive_gmm = AdaptiveGMMPolicy(
            base_update_freq=100,
            momentum=config["momentum"],
            warmup_iters=500,
            adaptive_freq=True
        )
        
        thresholds = []
        variances = []
        
        for iteration in range(0, 1000, 100):
            conf_tensor = torch.tensor(confidence_values, dtype=torch.float32)
            threshold, updated = adaptive_gmm.get_threshold(conf_tensor, iteration)
            thresholds.append(threshold)
            
            if len(thresholds) > 1:
                variance = abs(thresholds[-1] - thresholds[-2])
                variances.append(variance)
        
        # 计算稳定性指标
        avg_variance = np.mean(variances) if variances else 0
        final_threshold = thresholds[-1] if thresholds else 0
        
        results[config["name"]] = {
            "thresholds": thresholds,
            "avg_variance": avg_variance,
            "final_threshold": final_threshold
        }
        
        print(f"  Average Variance: {avg_variance:.6f}")
        print(f"  Final Threshold: {final_threshold:.4f}")
    
    return results

def suggest_improvements(results):
    """基于测试结果提出改进建议"""
    print("\n=== Improvement Suggestions ===")
    
    # 找到最稳定的配置
    best_config = min(results.items(), key=lambda x: x[1]["avg_variance"])
    print(f"Most stable configuration: {best_config[0]}")
    print(f"  Variance: {best_config[1]['avg_variance']:.6f}")
    print(f"  Final threshold: {best_config[1]['final_threshold']:.4f}")
    
    print("\nRecommended fixes:")
    print("1. Try lower momentum values (0.8-0.9 instead of 0.95)")
    print("2. Consider reducing update frequency to 50-75 iterations")
    print("3. Add threshold bounds to prevent extreme values")
    print("4. Implement confidence-aware momentum (lower momentum when confidence changes rapidly)")

if __name__ == "__main__":
    print("GMM Debug Tool")
    print("==============")
    
    # 生成测试数据
    np.random.seed(42)
    confidence_values = np.random.beta(2, 2, 10000)
    
    print(f"Using {len(confidence_values)} confidence values")
    print(f"Confidence range: [{confidence_values.min():.3f}, {confidence_values.max():.3f}]")
    print(f"Confidence mean: {confidence_values.mean():.3f}")
    
    # 分析GMM行为
    adaptive_thresholds, original_thresholds, update_frequencies = analyze_gmm_behavior(confidence_values)
    
    # 可视化结果
    visualize_debug_results(adaptive_thresholds, original_thresholds, update_frequencies)
    
    # 测试不同参数
    results = test_different_parameters()
    
    # 提出改进建议
    suggest_improvements(results) 