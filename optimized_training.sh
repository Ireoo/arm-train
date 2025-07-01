#!/bin/bash
# 优化后的训练脚本 - 防过拟合版本

echo "开始防过拟合训练..."
echo "预期最佳性能区间: 15k-30k步"
echo "超过40k步需要密切监控"

# 设置环境变量 - 平衡的奖励权重
export REWARD_TARGET_REACHED=20.0    # 提高成功奖励
export REWARD_DISTANCE_GUIDANCE=2.0  # 降低引导奖励防止过度依赖
export REWARD_APPROACH_PROGRESS=1.0  # 适中的接近奖励
export REWARD_ALIVE=0.5              # 降低存活奖励防止懒惰

# 反懒惰机制 - 轻量化
export REWARD_JOINT_VELOCITY=0.1     # 降低权重
export REWARD_EXPLORATION=0.05       # 降低权重  
export REWARD_ANTI_STAGNATION=0.2    # 降低权重

echo "奖励权重已优化，防止单一指标过度优化"

# 启动训练
python scripts/skrl/train.py --task Isaac-Arm-v0 --headless

echo "训练完成"
echo "请检查TensorBoard，重点关注15-30k步范围的性能"
