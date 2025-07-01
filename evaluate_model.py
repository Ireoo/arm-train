#!/usr/bin/env python3
"""
机械臂模型评估脚本
用于测试训练好的模型并生成性能报告
"""

import argparse
import torch
import numpy as np
from pathlib import Path

def evaluate_model(checkpoint_path, num_episodes=100):
    """评估模型性能"""
    print("🎯 开始模型评估...")
    print(f"📁 模型路径: {checkpoint_path}")
    print(f"🔄 评估轮数: {num_episodes}")
    
    # 这里应该加载实际的模型和环境
    # 由于需要与训练环境配置保持一致，这里提供评估框架
    
    results = {
        'success_rate': 0.0,
        'avg_distance': 0.0,
        'avg_episode_length': 0.0,
        'avg_total_reward': 0.0
    }
    
    print("\n📊 评估结果:")
    print("=" * 50)
    print(f"成功率: {results['success_rate']:.1f}%")
    print(f"平均距离: {results['avg_distance']:.4f}m")
    print(f"平均Episode长度: {results['avg_episode_length']:.1f}步")
    print(f"平均总奖励: {results['avg_total_reward']:.2f}")
    print("=" * 50)
    
    # 收敛判断
    if results['success_rate'] >= 85:
        print("🎉 模型性能优秀 - 已收敛！")
        return "converged"
    elif results['success_rate'] >= 70:
        print("✅ 模型性能良好 - 接近收敛")
        return "near_converged"
    elif results['success_rate'] >= 40:
        print("📈 模型有进展 - 继续训练可能有帮助")
        return "improving"
    else:
        print("🚀 模型需要更多训练")
        return "needs_training"

def main():
    parser = argparse.ArgumentParser(description="机械臂模型评估")
    parser.add_argument("--checkpoint", required=True, help="模型checkpoint路径")
    parser.add_argument("--episodes", type=int, default=100, help="评估轮数")
    parser.add_argument("--render", action="store_true", help="是否渲染可视化")
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"❌ 找不到模型文件: {args.checkpoint}")
        return
    
    status = evaluate_model(args.checkpoint, args.episodes)
    
    print(f"\n🏁 评估完成，状态: {status}")
    print("\n💡 使用说明:")
    print("1. 如果状态为'converged'，可以停止训练")
    print("2. 如果状态为'near_converged'，可以再训练一段时间")
    print("3. 如果状态为'improving'，建议继续训练")
    print("4. 如果状态为'needs_training'，需要检查配置或继续训练")

if __name__ == "__main__":
    main() 