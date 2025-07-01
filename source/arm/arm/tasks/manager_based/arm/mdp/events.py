# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import reset_root_state_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 全局标志，确保目标位置只在启动时初始化一次
_target_initialized = False


def initialize_target_position_on_startup(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    target_cfg: SceneEntityCfg,
) -> None:
    """在训练启动时初始化目标位置（仅执行一次）。
    
    Args:
        env: 环境实例
        env_ids: 环境ID张量
        target_cfg: 目标标记的配置
    """
    global _target_initialized
    
    # 如果已经初始化过，直接返回
    if _target_initialized:
        return
    # 定义目标位置的随机范围
    pose_range = {
        "x": (-0.3, 0.3),   # x轴范围: ±30cm
        "y": (-0.3, 0.3),   # y轴范围: ±30cm
        "z": (0.1, 0.3),    # z轴范围: 10cm到30cm高度
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
    
    try:
        # 使用reset_root_state_uniform函数来设置目标位置
        reset_root_state_uniform(
            env, 
            env_ids=env_ids,
            asset_cfg=target_cfg,
            pose_range=pose_range, 
            velocity_range=velocity_range
        )
        
        # 获取目标标记对象并打印位置信息
        target_marker = env.scene[target_cfg.name]
        target_pos = target_marker.data.root_pos_w[:, :3]
        
        print(f"✓ 成功初始化了 {len(env_ids)} 个环境的目标位置")
        print(f"目标位置范围: x∈{pose_range['x']}, y∈{pose_range['y']}, z∈{pose_range['z']}")
        
        # 显示前几个环境的目标位置样本
        sample_count = min(3, len(env_ids))
        for i in range(sample_count):
            sample_pos = target_pos[env_ids[i]].cpu().numpy()
            print(f"  环境#{env_ids[i].item()}: 目标位置=({sample_pos[0]:.3f}, {sample_pos[1]:.3f}, {sample_pos[2]:.3f})")
        
        # 设置标志，表示已经初始化过
        _target_initialized = True
            
    except Exception as e:
        print(f"❌ 初始化目标位置时出错: {e}")
        print(f"target_cfg类型: {type(target_cfg)}, target_cfg名称: {target_cfg.name if hasattr(target_cfg, 'name') else '未知'}") 