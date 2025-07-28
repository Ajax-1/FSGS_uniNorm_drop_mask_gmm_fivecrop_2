#!/usr/bin/env python3
"""
训练稳定性分析脚本
用于对比原始GMM和自适应GMM策略的训练稳定性
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

def extract_tensorboard_data(logdir, tag):
    """从TensorBoard日志中提取数据"""
    ea = EventAccumulator(logdir)
    ea.Reload()
    
    try:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        return np.array(steps), np.array(values)
    except KeyError:
        print(f"Warning: Tag '{tag}' not found in {logdir}")
        return np.array([]), np.array([])

def calculate_stability_metrics(values, window_size=100):
    """计算稳定性指标"""
    if len(values) < window_size:
        return {}
    
    # 计算滑动窗口的方差
    rolling_variance = []
    for i in range(window_size, len(values)):
        window = values[i-window_size:i]
        rolling_variance.append(np.var(window))
    
    # 计算稳定性指标
    mean_variance = np.mean(rolling_variance)
    variance_of_variance = np.var(rolling_variance)
    
    # 计算损失的平滑度（相邻点的差异）
    if len(values) > 1:
        diff = np.diff(values)
        smoothness = np.mean(np.abs(diff))
    else:
        smoothness = 0
    
    return {
        'mean_variance': mean_variance,
        'variance_of_variance': variance_of_variance,
        'smoothness': smoothness,
        'final_value': values[-1] if len(values) > 0 else 0
    }

def analyze_experiments(baseline_dir, adaptive_dir, output_dir):
    """分析实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 要分析的指标
    metrics_to_analyze = [
        'train/total_loss',
        'train/l1_loss', 
        'train/ssim_loss',
        'train/gmm_threshold',
        'train/gmm_variance'
    ]
    
    results = {}
    
    for metric in metrics_to_analyze:
        print(f"分析指标: {metric}")
        
        # 提取数据
        baseline_steps, baseline_values = extract_tensorboard_data(baseline_dir, metric)
        adaptive_steps, adaptive_values = extract_tensorboard_data(adaptive_dir, metric)
        
        if len(baseline_values) == 0 or len(adaptive_values) == 0:
            continue
            
        # 计算稳定性指标
        baseline_metrics = calculate_stability_metrics(baseline_values)
        adaptive_metrics = calculate_stability_metrics(adaptive_values)
        
        results[metric] = {
            'baseline': baseline_metrics,
            'adaptive': adaptive_metrics
        }
        
        # 绘制对比图
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(baseline_steps, baseline_values, label='Baseline GMM', alpha=0.7)
        plt.plot(adaptive_steps, adaptive_values, label='Adaptive GMM', alpha=0.7)
        plt.title(f'{metric} - 训练曲线对比')
        plt.xlabel('Iteration')
        plt.ylabel(metric.split('/')[-1])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 计算改进百分比
        if baseline_metrics['mean_variance'] > 0:
            variance_improvement = ((baseline_metrics['mean_variance'] - adaptive_metrics['mean_variance']) / 
                                  baseline_metrics['mean_variance'] * 100)
        else:
            variance_improvement = 0
            
        if baseline_metrics['smoothness'] > 0:
            smoothness_improvement = ((baseline_metrics['smoothness'] - adaptive_metrics['smoothness']) / 
                                    baseline_metrics['smoothness'] * 100)
        else:
            smoothness_improvement = 0
        
        plt.subplot(1, 2, 2)
        categories = ['方差', '平滑度', '方差的方差']
        baseline_vals = [baseline_metrics['mean_variance'], 
                        baseline_metrics['smoothness'],
                        baseline_metrics['variance_of_variance']]
        adaptive_vals = [adaptive_metrics['mean_variance'], 
                        adaptive_metrics['smoothness'],
                        adaptive_metrics['variance_of_variance']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, baseline_vals, width, label='Baseline GMM', alpha=0.7)
        plt.bar(x + width/2, adaptive_vals, width, label='Adaptive GMM', alpha=0.7)
        plt.title(f'{metric} - 稳定性指标对比')
        plt.xlabel('稳定性指标')
        plt.ylabel('数值')
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.replace("/", "_")}_comparison.png'))
        plt.close()
        
        print(f"  方差改进: {variance_improvement:.2f}%")
        print(f"  平滑度改进: {smoothness_improvement:.2f}%")
    
    # 生成总结报告
    generate_summary_report(results, output_dir)
    
    return results

def generate_summary_report(results, output_dir):
    """生成总结报告"""
    report_path = os.path.join(output_dir, 'stability_analysis_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 自适应GMM训练稳定性分析报告\n\n")
        f.write("## 实验概述\n")
        f.write("本报告对比了原始GMM策略和自适应GMM策略的训练稳定性。\n\n")
        f.write("## 分析结果\n\n")
        
        for metric, data in results.items():
            f.write(f"### {metric}\n")
            baseline = data['baseline']
            adaptive = data['adaptive']
            
            f.write("| 指标 | Baseline GMM | Adaptive GMM | 改进率 |\n")
            f.write("|------|--------------|---------------|--------|\n")
            
            # 方差改进
            if baseline['mean_variance'] > 0:
                var_improvement = (baseline['mean_variance'] - adaptive['mean_variance']) / baseline['mean_variance'] * 100
            else:
                var_improvement = 0
            f.write(f"| 平均方差 | {baseline['mean_variance']:.6f} | {adaptive['mean_variance']:.6f} | {var_improvement:+.2f}% |\n")
            
            # 平滑度改进
            if baseline['smoothness'] > 0:
                smooth_improvement = (baseline['smoothness'] - adaptive['smoothness']) / baseline['smoothness'] * 100
            else:
                smooth_improvement = 0
            f.write(f"| 平滑度 | {baseline['smoothness']:.6f} | {adaptive['smoothness']:.6f} | {smooth_improvement:+.2f}% |\n")
            
            # 最终值
            f.write(f"| 最终值 | {baseline['final_value']:.6f} | {adaptive['final_value']:.6f} | - |\n")
            f.write("\n")
        
        f.write("## 结论\n")
        f.write("- 如果改进率为正值，说明自适应GMM策略效果更好\n")
        f.write("- 方差降低意味着训练更稳定\n")
        f.write("- 平滑度改善意味着损失曲线更平滑\n")

def main():
    parser = argparse.ArgumentParser(description='分析训练稳定性')
    parser.add_argument('--baseline_dir', required=True, help='基线实验TensorBoard日志目录')
    parser.add_argument('--adaptive_dir', required=True, help='自适应GMM实验TensorBoard日志目录')
    parser.add_argument('--output_dir', default='analysis_results', help='输出目录')
    
    args = parser.parse_args()
    
    results = analyze_experiments(args.baseline_dir, args.adaptive_dir, args.output_dir)
    print(f"分析完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 