# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import random

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.envs.mdp.events import reset_root_state_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def end_effector_position_l2(env: ManagerBasedRLEnv, target_position: list, asset_cfg: SceneEntityCfg, body_name: str = "arm_end") -> torch.Tensor:
    """Penalize end-effector position deviation from a target 3D position."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get end-effector position - shape: (num_envs, 3)
    ee_pos = asset.data.body_pos_w[:, asset.find_bodies(body_name)[0], :3]
    
    # Convert target position to tensor with correct shape (num_envs, 3)
    # Use the actual batch size from ee_pos
    num_envs = ee_pos.shape[0]
    target_pos = torch.tensor(target_position, dtype=torch.float32, device=env.device)
    target_pos = target_pos.unsqueeze(0).expand(num_envs, -1)  # Shape: (num_envs, 3)
    
    # compute the L2 distance to target - shape: (num_envs,)
    distance = torch.norm(ee_pos - target_pos, dim=1)
    return distance


def end_effector_position_to_marker_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, body_name: str = "arm_end") -> torch.Tensor:
    """Penalize end-effector position deviation from target marker position."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target_marker = env.scene[target_cfg.name]
    
    # Get the actual number of environments
    num_envs = asset.data.root_pos_w.shape[0]
    
    # Get end-effector position - shape: (num_envs, 3)
    try:
        body_indices = asset.find_bodies(body_name)
        if len(body_indices) == 0:
            print(f"Warning: Body '{body_name}' not found, using root position")
            ee_pos = asset.data.root_pos_w[:, :3]
        else:
            ee_pos = asset.data.body_pos_w[:, body_indices[0], :3]
            # Remove extra dimensions if present: (num_envs, 1, 3) -> (num_envs, 3)
            if ee_pos.dim() == 3 and ee_pos.shape[1] == 1:
                ee_pos = ee_pos.squeeze(1)
    except Exception as e:
        print(f"Error finding body '{body_name}': {e}")
        ee_pos = asset.data.root_pos_w[:, :3]
    
    # Get target marker position - shape: (num_envs, 3)
    target_pos = target_marker.data.root_pos_w[:, :3]
    
    # Ensure both tensors have the correct shape
    if ee_pos.shape[0] != num_envs or ee_pos.shape[1] != 3:
        print(f"Error: ee_pos has wrong shape {ee_pos.shape}, expected ({num_envs}, 3)")
        return torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    
    if target_pos.shape[0] != num_envs or target_pos.shape[1] != 3:
        print(f"Error: target_pos has wrong shape {target_pos.shape}, expected ({num_envs}, 3)")
        return torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    
    # compute the L2 distance to target - shape: (num_envs,)
    try:
        distance = torch.norm(ee_pos - target_pos, dim=1)
        
        # Final safety check
        if distance.shape[0] != num_envs:
            print(f"Error: distance shape {distance.shape} != expected ({num_envs},)")
            return torch.zeros(num_envs, device=env.device, dtype=torch.float32)
              
        return distance
        
    except Exception as e:
        print(f"Error computing distance: {e}")
        return torch.zeros(num_envs, device=env.device, dtype=torch.float32)


def target_reached_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, body_name: str = "arm_end") -> torch.Tensor:
    """给予成功到达目标的奖励加成。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target_marker = env.scene[target_cfg.name]
    
    # Get the actual number of environments
    num_envs = asset.data.root_pos_w.shape[0]
    
    # Get end-effector position - shape: (num_envs, 3)
    try:
        body_indices = asset.find_bodies(body_name)
        if len(body_indices) == 0:
            ee_pos = asset.data.root_pos_w[:, :3]
        else:
            ee_pos = asset.data.body_pos_w[:, body_indices[0], :3]
            if ee_pos.dim() == 3 and ee_pos.shape[1] == 1:
                ee_pos = ee_pos.squeeze(1)
    except Exception as e:
        ee_pos = asset.data.root_pos_w[:, :3]
    
    # Get target marker position - shape: (num_envs, 3)
    target_pos = target_marker.data.root_pos_w[:, :3]
    
    # compute the L2 distance to target - shape: (num_envs,)
    distance = torch.norm(ee_pos - target_pos, dim=1)
    
    # 课程学习：动态调整成功阈值，从容易到困难
    if hasattr(env, '_curriculum_step'):
        env._curriculum_step += 1
    else:
        env._curriculum_step = 0
    
    # 渐进式成功阈值：从8cm->5cm->3cm->2cm
    if env._curriculum_step < 20000:
        success_threshold = 0.08  # 前20k步：8cm
    elif env._curriculum_step < 40000:
        success_threshold = 0.05  # 20-40k步：5cm  
    elif env._curriculum_step < 60000:
        success_threshold = 0.03  # 40-60k步：3cm
    else:
        success_threshold = 0.02  # 60k+步：2cm
    
    success_bonus = torch.where(distance < success_threshold, 
                               torch.tensor(300.0, device=env.device, dtype=torch.float32),  # 增加成功奖励
                               torch.tensor(0.0, device=env.device, dtype=torch.float32))
    
    collision_detected = distance < success_threshold
        
    # 如果检测到碰撞，仅打印调试信息
    if torch.any(collision_detected):
        collision_count = torch.sum(collision_detected).item()
        collision_indices = torch.where(collision_detected)[0]
        if len(collision_indices) > 0:
            sample_idx = collision_indices[0].item()
            # sample_ee_pos = ee_pos[sample_idx].cpu().numpy()
            # sample_target_pos = target_pos[sample_idx].cpu().numpy()
            sample_distance = distance[sample_idx].item()
            # print(f"目标到达: 检测到{collision_count}个成功到达事件, 奖励{success_bonus[sample_idx]:.4f}, 样本成功 #{sample_idx}: 距离={sample_distance:.4f}m")
            update_target_marker(env, asset_cfg, target_cfg, collision_indices)
    
    return success_bonus


def distance_guidance_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, body_name: str = "arm_end") -> torch.Tensor:
    """基于距离的引导奖励，距离越近奖励越高。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target_marker = env.scene[target_cfg.name]
    
    # Get the actual number of environments
    num_envs = asset.data.root_pos_w.shape[0]
    
    # Get end-effector position - shape: (num_envs, 3)
    try:
        body_indices = asset.find_bodies(body_name)
        if len(body_indices) == 0:
            ee_pos = asset.data.root_pos_w[:, :3]
        else:
            ee_pos = asset.data.body_pos_w[:, body_indices[0], :3]
            if ee_pos.dim() == 3 and ee_pos.shape[1] == 1:
                ee_pos = ee_pos.squeeze(1)
    except Exception as e:
        ee_pos = asset.data.root_pos_w[:, :3]
    
    # Get target marker position - shape: (num_envs, 3)
    target_pos = target_marker.data.root_pos_w[:, :3]
    
    # compute the L2 distance to target - shape: (num_envs,)
    distance = torch.norm(ee_pos - target_pos, dim=1)
    
    # 基于距离的渐进奖励：距离越近，奖励越高
    # 使用指数衰减函数：reward = exp(-distance * scale)
    guidance_reward = torch.exp(-distance * 3.0)  # 降低scale使远距离也有奖励
    
    # 添加反懒惰机制：奖励向目标移动的行为
    if hasattr(env, '_prev_distance'):
        # 如果距离在缩短，给予额外奖励
        distance_improvement = env._prev_distance - distance
        improvement_bonus = torch.clamp(distance_improvement * 100.0, -5.0, 10.0)  # 限制奖励范围
        guidance_reward += improvement_bonus
    
    env._prev_distance = distance.clone()
    
    return guidance_reward


def approach_progress_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, body_name: str = "arm_end") -> torch.Tensor:
    """奖励机械臂接近目标的进步。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target_marker = env.scene[target_cfg.name]
    
    # Get the actual number of environments
    num_envs = asset.data.root_pos_w.shape[0]
    
    # Get end-effector position - shape: (num_envs, 3)
    try:
        body_indices = asset.find_bodies(body_name)
        if len(body_indices) == 0:
            ee_pos = asset.data.root_pos_w[:, :3]
        else:
            ee_pos = asset.data.body_pos_w[:, body_indices[0], :3]
            if ee_pos.dim() == 3 and ee_pos.shape[1] == 1:
                ee_pos = ee_pos.squeeze(1)
    except Exception as e:
        ee_pos = asset.data.root_pos_w[:, :3]
    
    # Get target marker position - shape: (num_envs, 3)
    target_pos = target_marker.data.root_pos_w[:, :3]
    
    # compute the L2 distance to target - shape: (num_envs,)
    distance = torch.norm(ee_pos - target_pos, dim=1)
    
    # 创建一个更激进的接近奖励
    # 当距离小于某个阈值时给予大奖励
    close_threshold = 0.1  # 10cm内给予接近奖励
    very_close_threshold = 0.05  # 5cm内给予更高奖励
    
    approach_reward = torch.zeros_like(distance)
    
    # 10cm内给予基础接近奖励
    close_mask = distance < close_threshold
    approach_reward[close_mask] += 10.0
    
    # 5cm内给予额外奖励
    very_close_mask = distance < very_close_threshold
    approach_reward[very_close_mask] += 20.0
    
    # 2cm内给予巨大奖励
    success_mask = distance < 0.02
    approach_reward[success_mask] += 50.0
    
    return approach_reward


def convergence_monitor(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, body_name: str = "arm_end") -> torch.Tensor:
    """监控训练收敛状态的关键指标。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target_marker = env.scene[target_cfg.name]
    
    # Get the actual number of environments
    num_envs = asset.data.root_pos_w.shape[0]
    
    # Get end-effector position
    try:
        body_indices = asset.find_bodies(body_name)
        if len(body_indices) == 0:
            ee_pos = asset.data.root_pos_w[:, :3]
        else:
            ee_pos = asset.data.body_pos_w[:, body_indices[0], :3]
            if ee_pos.dim() == 3 and ee_pos.shape[1] == 1:
                ee_pos = ee_pos.squeeze(1)
    except Exception as e:
        ee_pos = asset.data.root_pos_w[:, :3]
    
    # Get target position
    target_pos = target_marker.data.root_pos_w[:, :3]
    
    # Calculate distances
    distance = torch.norm(ee_pos - target_pos, dim=1)
    
    # 收敛指标统计
    success_count = torch.sum(distance < 0.05).item()  # 5cm内成功数
    close_count = torch.sum(distance < 0.1).item()    # 10cm内接近数
    avg_distance = torch.mean(distance).item()         # 平均距离
    
    # 每100步打印一次收敛统计
    if hasattr(env, '_convergence_step_counter'):
        env._convergence_step_counter += 1
    else:
        env._convergence_step_counter = 0
    
    if env._convergence_step_counter % 1000 == 0:
        success_rate = success_count / num_envs * 100
        close_rate = close_count / num_envs * 100
        
        print(f"=== 收敛监控 (Step {env._convergence_step_counter}) ===")
        print(f"成功率 (5cm内): {success_rate:.1f}% ({success_count}/{num_envs})")
        print(f"接近率 (10cm内): {close_rate:.1f}% ({close_count}/{num_envs})")
        print(f"平均距离: {avg_distance:.4f}m")
        
        # 收敛判断逻辑
        if success_rate >= 80:
            print("🎉 收敛判断: 任务成功率达标 (>=80%)")
        elif success_rate >= 50:
            print("🔄 收敛判断: 接近收敛 (50-80%)")
        elif close_rate >= 30:
            print("📈 收敛判断: 学习进展良好 (>30%接近)")
        else:
            print("🚀 收敛判断: 继续训练中...")
        
        print("=" * 50)
    
    return torch.zeros(num_envs, device=env.device, dtype=torch.float32)


def termination_monitor(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """监控终止条件，帮助调试。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the actual number of environments
    num_envs = asset.data.root_pos_w.shape[0]
    
    # 检查关节位置是否超出边界
    joint_pos = asset.data.joint_pos
    joint_bounds_violations = 0
    
    # 检查主要关节（joint_2-7）
    for i in range(1, 7):  # joint_2 to joint_7 (0-indexed: 1-6)
        if i < joint_pos.shape[1]:
            joint_i_pos = joint_pos[:, i]
            out_of_bounds = torch.logical_or(joint_i_pos < -3.0 * torch.pi, joint_i_pos > 3.0 * torch.pi)
            if torch.any(out_of_bounds):
                violations = torch.sum(out_of_bounds).item()
                joint_bounds_violations += violations
                if violations > 0:
                    print(f"关节 {i+1} 边界违规: {violations} 个环境")
    
    # 检查末端执行器关节（joint_1, joint_8）
    for i in [0, 7]:  # joint_1 and joint_8 (0-indexed: 0, 7)
        if i < joint_pos.shape[1]:
            joint_i_pos = joint_pos[:, i]
            out_of_bounds = torch.logical_or(joint_i_pos < -torch.pi, joint_i_pos > 3.0 * torch.pi)
            if torch.any(out_of_bounds):
                violations = torch.sum(out_of_bounds).item()
                joint_bounds_violations += violations
                if violations > 0:
                    print(f"末端关节 {i+1} 边界违规: {violations} 个环境")
    
    if joint_bounds_violations > 0:
        print(f"总计关节边界违规: {joint_bounds_violations} 次")
    
    return torch.zeros(num_envs, device=env.device, dtype=torch.float32)


def update_target_marker(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, collision_indices):
    # 定义位置范围和速度范围
    pose_range = {
        "x": (-0.3, 0.3),  # x轴范围: 圆心±89cm (考虑89cm最大半径)
        "y": (-0.3, 0.3),  # y轴范围: 圆心±89cm
        "z": (0.1, 0.3),   # z轴范围: 基准位置向上15cm附近
        "roll": (0.0, 0.0),  # 保持旋转为0
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    
    # 正确调用reset_root_state_uniform函数
    # 根据错误信息，target_cfg不是有效的索引参数
    # 需要找到正确的目标对象索引或ID来重置状态
    try:
        # 直接使用target_cfg来获取目标对象并重置其状态
        # target_cfg是SceneEntityCfg对象，包含了目标标记的信息
        reset_root_state_uniform(
            env, 
            env_ids=collision_indices,  # 只重置碰撞的环境
            asset_cfg=target_cfg,  # 使用传入的目标配置
            pose_range=pose_range, 
            velocity_range=velocity_range
        )
        # print(f"成功重置了 {len(collision_indices)} 个环境的目标位置")
    except Exception as e:
        print(f"重置目标位置时出错: {e}")
        print(f"target_cfg类型: {type(target_cfg)}, target_cfg名称: {target_cfg.name if hasattr(target_cfg, 'name') else '未知'}")
    # target_cfg.init_state.pos = (random.uniform(-0.89, 0.89), random.uniform(-0.89, 0.89), random.uniform(0.1, 0.5))


def joint_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励关节运动，防止懒惰行为。"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取关节速度
    joint_vel = asset.data.joint_vel
    
    # 计算关节速度的L2范数（总运动量）
    velocity_magnitude = torch.norm(joint_vel, dim=1)
    
    # 奖励适度的运动，但惩罚过度的运动
    # 目标是鼓励有目的的运动而不是随机抖动
    optimal_velocity = 2.0  # rad/s，适度的关节运动速度
    velocity_reward = torch.exp(-torch.abs(velocity_magnitude - optimal_velocity) / optimal_velocity)
    
    # 基础运动奖励：只要在动就给奖励
    movement_bonus = torch.clamp(velocity_magnitude * 0.5, 0.0, 2.0)
    
    return velocity_reward + movement_bonus


def exploration_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, body_name: str = "arm_end") -> torch.Tensor:
    """奖励探索新区域，防止陷入局部区域。"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the actual number of environments
    num_envs = asset.data.root_pos_w.shape[0]
    
    # Get end-effector position
    try:
        body_indices = asset.find_bodies(body_name)
        if len(body_indices) == 0:
            ee_pos = asset.data.root_pos_w[:, :3]
        else:
            ee_pos = asset.data.body_pos_w[:, body_indices[0], :3]
            if ee_pos.dim() == 3 and ee_pos.shape[1] == 1:
                ee_pos = ee_pos.squeeze(1)
    except Exception as e:
        ee_pos = asset.data.root_pos_w[:, :3]
    
    # 维护探索历史（简化版本）
    if not hasattr(env, '_exploration_history'):
        env._exploration_history = []
        env._exploration_step = 0
    
    env._exploration_step += 1
    
    # 每100步记录一次位置
    if env._exploration_step % 100 == 0:
        current_positions = ee_pos.cpu().numpy()
        env._exploration_history.append(current_positions)
        
        # 只保留最近的50个记录
        if len(env._exploration_history) > 50:
            env._exploration_history.pop(0)
    
    # 计算探索奖励：与历史位置的平均距离
    exploration_reward = torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    
    if len(env._exploration_history) > 1:
        for i, historical_pos in enumerate(env._exploration_history[-10:]):  # 检查最近10个历史位置
            historical_tensor = torch.tensor(historical_pos, device=env.device, dtype=torch.float32)
            distances = torch.norm(ee_pos - historical_tensor, dim=1)
            # 奖励与历史位置的距离（鼓励探索新区域）
            exploration_reward += torch.clamp(distances * 2.0, 0.0, 5.0)
        
        exploration_reward /= min(len(env._exploration_history), 10)  # 平均化
    
    return exploration_reward


def anti_stagnation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, body_name: str = "arm_end") -> torch.Tensor:
    """反停滞奖励：检测并惩罚长时间不改善的行为。"""
    asset: Articulation = env.scene[asset_cfg.name]
    target_marker = env.scene[target_cfg.name]
    
    # Get the actual number of environments
    num_envs = asset.data.root_pos_w.shape[0]
    
    # Get end-effector position
    try:
        body_indices = asset.find_bodies(body_name)
        if len(body_indices) == 0:
            ee_pos = asset.data.root_pos_w[:, :3]
        else:
            ee_pos = asset.data.body_pos_w[:, body_indices[0], :3]
            if ee_pos.dim() == 3 and ee_pos.shape[1] == 1:
                ee_pos = ee_pos.squeeze(1)
    except Exception as e:
        ee_pos = asset.data.root_pos_w[:, :3]
    
    # Get target position
    target_pos = target_marker.data.root_pos_w[:, :3]
    distance = torch.norm(ee_pos - target_pos, dim=1)
    
    # 维护性能历史
    if not hasattr(env, '_performance_history'):
        env._performance_history = []
        env._stagnation_counter = torch.zeros(num_envs, device=env.device)
    
    # 每50步检查一次是否有改善
    if len(env._performance_history) == 0:
        env._performance_history.append(distance.clone())
        return torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    
    # 检查是否有改善
    last_distance = env._performance_history[-1]
    improvement = last_distance - distance  # 正值表示改善
    
    # 更新停滞计数器
    no_improvement_mask = improvement < 0.001  # 改善小于1mm视为没有改善
    env._stagnation_counter[no_improvement_mask] += 1
    env._stagnation_counter[~no_improvement_mask] = 0  # 有改善则重置计数器
    
    # 计算反停滞奖励
    anti_stagnation_reward = torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    
    # 惩罚长时间停滞（超过500步没有改善）
    stagnation_penalty = torch.where(env._stagnation_counter > 500,
                                   -torch.log(env._stagnation_counter / 500.0),  # 随时间增加的惩罚
                                   torch.tensor(0.0, device=env.device))
    
    # 奖励最近的改善
    recent_improvement_bonus = torch.clamp(improvement * 50.0, -2.0, 5.0)
    
    anti_stagnation_reward = recent_improvement_bonus + stagnation_penalty
    
    # 更新历史（保留最近的记录）
    env._performance_history.append(distance.clone())
    if len(env._performance_history) > 100:  # 只保留最近100个记录
        env._performance_history.pop(0)
    
    return anti_stagnation_reward