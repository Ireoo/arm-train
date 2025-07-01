# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the position of the specified body in world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body indices - find_bodies expects individual body names, not a list
    # asset_cfg.body_names is a list, so we need to handle each body name
    if len(asset_cfg.body_names) == 1:
        # Single body case
        body_name = asset_cfg.body_names[0]
        body_indices = asset.find_bodies(body_name)
        # Get body position and ensure shape is (num_envs, 3)
        body_positions = asset.data.body_pos_w[:, body_indices[0], :3]
        
        # Remove extra dimensions if present: (num_envs, 1, 3) -> (num_envs, 3)
        if body_positions.dim() == 3 and body_positions.shape[1] == 1:
            body_positions = body_positions.squeeze(1)
        # If somehow we get (3,) instead of (num_envs, 3), expand it
        elif body_positions.dim() == 1 and body_positions.shape[0] == 3:
            # Get the actual number of environments from the asset data
            num_envs = asset.data.body_pos_w.shape[0]
            body_positions = body_positions.unsqueeze(0).expand(num_envs, -1)
            
    else:
        # Multiple bodies case
        all_positions = []
        for body_name in asset_cfg.body_names:
            body_indices = asset.find_bodies(body_name)
            body_pos = asset.data.body_pos_w[:, body_indices[0], :3]
            # Remove extra dimensions if present: (num_envs, 1, 3) -> (num_envs, 3)
            if body_pos.dim() == 3 and body_pos.shape[1] == 1:
                body_pos = body_pos.squeeze(1)
            all_positions.append(body_pos)
        # Concatenate along the last dimension: (num_envs, num_bodies * 3)
        body_positions = torch.cat(all_positions, dim=-1)
    
    return body_positions


def root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the position of the asset's root in world frame."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    
    # Get root position in world frame - shape: (num_envs, 3)
    root_position = asset.data.root_pos_w[:, :3]
    
    return root_position 