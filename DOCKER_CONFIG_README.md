# ARM 训练 Docker 配置指南

本指南说明如何使用 docker-compose.yaml 中的环境变量来动态配置 ARM 训练参数，无需修改源代码。

## 🚀 快速开始

### 1. 启动后台训练
```bash
# 使用默认配置启动
docker-compose up -d

# 查看训练日志
docker-compose logs -f arm-train

# 停止训练
docker-compose down
```

### 2. 自定义配置
编辑 `docker-compose.yaml` 文件中的环境变量，然后重启容器：

```bash
# 修改配置后重启
docker-compose restart
```

## 📋 可配置参数

### 🔧 训练基础参数

| 环境变量 | 描述 | 默认值 | 示例 |
|---------|------|--------|------|
| `NUM_ENVS` | 并行环境数量 | `2048` | `4096` |
| `TIMESTEPS` | 总训练时间步数 | `100000` | `200000` |
| `EPISODE_LENGTH_S` | 每个episode长度(秒) | `1200` | `1800` |

### 🎯 奖励函数权重

| 环境变量 | 描述 | 默认值 | 建议范围 |
|---------|------|--------|----------|
| `REWARD_ALIVE` | 存活奖励权重 | `1.0` | `0.5-2.0` |
| `REWARD_TERMINATING` | 终止惩罚权重 | `-2.0` | `-5.0~-1.0` |
| `REWARD_END_EFFECTOR_POSITION` | 末端执行器位置奖励权重 | `-5.0` | `-10.0~-1.0` |
| `REWARD_JOINT_VEL` | 关节速度惩罚权重 | `-0.01` | `-0.1~-0.001` |
| `REWARD_JOINT_VEL_SMOOTH` | 关节速度平滑惩罚权重 | `-0.005` | `-0.05~-0.001` |

### 🧠 PPO 算法参数

| 环境变量 | 描述 | 默认值 | 建议范围 |
|---------|------|--------|----------|
| `LEARNING_RATE` | 学习率 | `0.0003` | `0.0001-0.001` |
| `ROLLOUTS` | 滚动缓冲区大小 | `64` | `32-128` |
| `LEARNING_EPOCHS` | 每次更新的学习轮数 | `10` | `5-20` |
| `MINI_BATCHES` | 小批次数量 | `16` | `8-32` |
| `DISCOUNT_FACTOR` | 折扣因子 | `0.99` | `0.95-0.999` |
| `ENTROPY_LOSS_SCALE` | 熵损失缩放 | `0.01` | `0.001-0.1` |
| `VALUE_LOSS_SCALE` | 价值损失缩放 | `2.0` | `1.0-5.0` |

## 🔧 配置示例

### 快速训练配置
```yaml
environment:
  - NUM_ENVS=1024                    # 减少环境数量
  - TIMESTEPS=50000                  # 减少训练步数
  - LEARNING_RATE=0.001              # 提高学习率
  - ROLLOUTS=32                      # 减少rollout
```

### 精细调优配置
```yaml
environment:
  - NUM_ENVS=4096                    # 增加环境数量
  - TIMESTEPS=500000                 # 增加训练步数
  - LEARNING_RATE=0.0001             # 降低学习率
  - REWARD_END_EFFECTOR_POSITION=-10.0  # 增强位置奖励
```

### 稳定训练配置
```yaml
environment:
  - LEARNING_RATE=0.0002             # 中等学习率
  - ROLLOUTS=64                      # 标准rollout
  - ENTROPY_LOSS_SCALE=0.05          # 增加探索
  - REWARD_JOINT_VEL=-0.02           # 增强平滑性
```

## 🛠️ 调参建议

### 训练速度优化
- **增加并行环境数量**: 提高 `NUM_ENVS` (需要更多GPU内存)
- **调整批次大小**: 增加 `MINI_BATCHES` (需要更多内存)
- **减少学习轮数**: 降低 `LEARNING_EPOCHS`

### 训练稳定性优化
- **降低学习率**: 减小 `LEARNING_RATE`
- **增加探索**: 提高 `ENTROPY_LOSS_SCALE`
- **平滑运动**: 增加 `REWARD_JOINT_VEL` 和 `REWARD_JOINT_VEL_SMOOTH` 的惩罚

### 任务性能优化
- **强化主要任务**: 增加 `REWARD_END_EFFECTOR_POSITION` 的权重 (负值的绝对值)
- **调整episode长度**: 根据任务复杂度调整 `EPISODE_LENGTH_S`

## 🔍 配置验证

使用提供的验证脚本检查配置是否正确加载：

```bash
# 在容器中运行验证脚本
docker-compose exec arm-train python /workspace/scripts/validate_config.py

# 或者在本地运行 (需要有 Python 环境)
python scripts/validate_config.py
```

## 📊 监控训练

### 查看实时日志
```bash
# 查看训练日志
docker-compose logs -f arm-train

# 查看最近的日志
docker-compose logs --tail=100 arm-train
```

### 检查训练进度
```bash
# 进入容器查看checkpoint
docker-compose exec arm-train ls -la /workspace/logs/skrl/arm/

# 查看TensorBoard日志
docker-compose exec arm-train find /workspace/logs -name "*.tfevents*"
```

## 🚨 故障排除

### 内存不足
- 减少 `NUM_ENVS` 参数
- 减少 `MINI_BATCHES` 参数
- 检查GPU内存使用情况

###训练不稳定
- 降低 `LEARNING_RATE`
- 增加 `ENTROPY_LOSS_SCALE`
- 调整奖励函数权重

### 训练速度慢
- 增加 `NUM_ENVS` (如果内存允许)
- 调整 `ROLLOUTS` 和 `MINI_BATCHES`
- 检查GPU利用率

## 📁 相关文件

- `docker-compose.yaml` - 主配置文件
- `source/arm/arm/tasks/manager_based/arm/arm_env_cfg.py` - 环境配置
- `scripts/skrl/train.py` - 训练脚本
- `scripts/validate_config.py` - 配置验证脚本

## 🎯 最佳实践

1. **逐步调参**: 每次只修改一个参数，观察效果
2. **保存配置**: 为不同的实验保存不同的 docker-compose 配置
3. **监控日志**: 定期检查训练日志和性能指标
4. **备份模型**: 定期备份训练好的checkpoint
5. **验证配置**: 使用验证脚本确保配置正确加载

---

## 📝 更新日志

- **2025-01-01**: 添加环境变量配置支持
- **2025-01-01**: 创建配置验证脚本
- **2025-01-01**: 添加详细的使用说明 