#!/usr/bin/env python3
"""
机械臂训练收敛检查器
用于分析训练指标并判断是否收敛
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

class ConvergenceChecker:
    def __init__(self, log_dir="outputs/arm"):
        self.log_dir = Path(log_dir)
        
    def check_tensorboard_logs(self):
        """检查TensorBoard日志中的关键指标"""
        print("🔍 检查TensorBoard日志...")
        
        # 寻找最新的训练运行
        runs = list(self.log_dir.glob("*"))
        if not runs:
            print("❌ 未找到训练日志目录")
            return False
            
        latest_run = max(runs, key=os.path.getctime)
        print(f"📁 最新训练目录: {latest_run}")
        
        return True
    
    def analyze_reward_convergence(self, reward_data):
        """分析奖励收敛情况"""
        if len(reward_data) < 100:
            return "数据不足"
        
        # 计算最近1000步的统计
        recent_data = reward_data[-1000:]
        mean_reward = np.mean(recent_data)
        std_reward = np.std(recent_data)
        
        # 计算变异系数 (CV = std/mean)
        cv = std_reward / abs(mean_reward) if mean_reward != 0 else float('inf')
        
        if cv < 0.1:
            return "🎉 已收敛 (CV < 0.1)"
        elif cv < 0.2:
            return "🔄 接近收敛 (CV < 0.2)"
        else:
            return "🚀 继续训练 (CV >= 0.2)"
    
    def check_success_rate_trend(self, success_data):
        """检查成功率趋势"""
        if len(success_data) < 50:
            return "数据不足"
        
        recent_success = success_data[-100:]
        success_rate = np.mean(recent_success) * 100
        
        if success_rate >= 80:
            return f"🎉 成功率达标: {success_rate:.1f}%"
        elif success_rate >= 50:
            return f"🔄 成功率良好: {success_rate:.1f}%"
        elif success_rate >= 20:
            return f"📈 成功率改善: {success_rate:.1f}%"
        else:
            return f"🚀 成功率较低: {success_rate:.1f}%"
    
    def generate_convergence_report(self):
        """生成收敛报告"""
        print("=" * 60)
        print("🎯 机械臂训练收敛报告")
        print("=" * 60)
        
        # 检查基本信息
        self.check_tensorboard_logs()
        
        print("\n📊 收敛判断准则:")
        print("1. 成功率 (5cm内到达目标):")
        print("   - 已收敛: ≥80%")
        print("   - 接近收敛: 50-80%")
        print("   - 继续训练: <50%")
        
        print("\n2. 奖励稳定性:")
        print("   - 已收敛: 变异系数CV <0.1")
        print("   - 接近收敛: CV 0.1-0.2")
        print("   - 继续训练: CV >0.2")
        
        print("\n3. Episode长度:")
        print("   - 稳定在目标范围: 1000-3000步")
        print("   - 不再大幅波动")
        
        print("\n4. 关键监控指标:")
        print("   - target_reached: 应持续>0")
        print("   - approach_progress: 应稳定在高值")
        print("   - distance_guidance: 应保持增长趋势")
        print("   - end_effector_position: 应趋向小的负值")
        
        print("\n🔧 如何使用:")
        print("1. 观察训练日志中的收敛监控输出")
        print("2. 在TensorBoard中查看奖励曲线")
        print("3. 当成功率稳定在80%以上时可考虑停止训练")
        print("4. 保存表现最好的checkpoint")
        
        print("\n💡 提前停止建议:")
        print("- 成功率连续500步保持>85%")
        print("- 总奖励连续1000步变异系数<0.05")
        print("- 平均到达距离稳定在<3cm")
        
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="机械臂训练收敛检查器")
    parser.add_argument("--log-dir", default="outputs/arm", help="训练日志目录")
    args = parser.parse_args()
    
    checker = ConvergenceChecker(args.log_dir)
    checker.generate_convergence_report()

if __name__ == "__main__":
    main() 