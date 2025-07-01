# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from isaaclab.actuators import ImplicitActuatorCfg

##
# Scene definition
##

CARTPOLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="./source/arm.usd",
    ),
    actuators={
        "joint_effort": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=15.0,  # 增加力矩限制，提供足够的力矩
            velocity_limit=2.0,  # 适当增加速度限制
            stiffness=80.0,  # 降低刚度以减少抖动
            damping=15.0,  # 增加阻尼以提供更好的稳定性
        )
    },
)

# 计算球形范围内的坐标
# 最小半径: 0.3米
# 最大半径: 1.2*7+5 = 13.4米
# 使用球坐标系转换为笛卡尔坐标系
# 使用与rewards.py中相同的坐标范围
# target_pos_range = ((-0.89, 0.89), (-0.89, 0.89), (0.1, 0.5))
# 随机选择一个值
import random
# target_pos_range = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(0.1, 0.3))

# print("目标位置:", target_pos_range)


@configclass
class ArmSceneCfg(InteractiveSceneCfg):
    """Configuration for a robotic arm scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10000.0, 10000.0)),
    )

    # robot
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 简单的目标球体用于可视化
    target_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target_marker",
        spawn=sim_utils.SphereCfg(
            radius=0.05,  # 增大球体半径，提高可见性
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # 设为运动学物体，不会因物理作用而移动
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),  # 质量为0
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,  # 禁用碰撞
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # 鲜艳的红色
                emissive_color=(0.3, 0.0, 0.0),  # 添加发光效果，增强可见性
                metallic=0.0,
                roughness=0.3,  # 降低粗糙度，增加反光
                opacity=1.0,  # 确保完全不透明
            ),
            activate_contact_sensors=True,  # 禁用接触传感器
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(0.1, 0.3))),  # 统一的起始位置
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["joint_[1-8]"], scale=100)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # end-effector position observation
        end_effector_pos = ObsTerm(func=mdp.body_pos_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=["arm_end"])})
        # target marker position (dynamic)
        target_position = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("target_marker")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # 训练启动时初始化目标位置（仅执行一次）
    initialize_target_position = EventTerm(
        func=mdp.initialize_target_position_on_startup,
        mode="reset",  # 在环境重置时执行，但函数内部会确保只在第一次执行
        params={
            "target_cfg": SceneEntityCfg("target_marker"),
        },
    )

    # reset - 机械臂关节重置，但不重置小球位置
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[2-7]"]),
            "position_range": (-math.pi / 4, math.pi / 4),  # 进一步缩小初始范围，避免边界问题
            "velocity_range": (-0.001, 0.001),  # 减少初始速度，避免不稳定
        },
    )

    # reset_arm_joints_2 = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_2","joint_4","joint_5"]),
    #         "position_range": (0, 1.5 * math.pi / 2),
    #         "velocity_range": (-0.001, 0.001),
    #     },
    # )

    # reset_arm_joints_3 = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_2"]),
    #         "position_range": (0, 0.5 * math.pi),
    #         "velocity_range": (-0.001, 0.001),
    #     },
    # )

    reset_end_effector_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_1","joint_8"]),
            "position_range": (math.pi / 2, math.pi),  # 更保守的中间位置
            "velocity_range": (-0.0001, 0.0001),
        },
    )

    # 更新目标标记位置的事件
    # update_target_marker = EventTerm(
    #     func=mdp.update_target_on_reach,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "target_cfg": SceneEntityCfg("target_marker"),
    #         "body_name": "arm_end",
    #         "threshold": 0.05,  # 5厘米的阈值，确保真正碰到
    #         "pose_range": {
    #             "x": (-0.39, 1.39),
    #             "y": (-0.89, 0.89),
    #             "z": (0.1, 0.5)
    #         }
    #     },
    #     interval_range_s=(0.1, 0.5),  # 每0.1到0.5秒检查一次
    #     is_global_time=False,
    #     min_step_count_between_reset=10,  # 至少10步后才开始检查
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward  
    alive = RewTerm(func=mdp.is_alive, weight=float(os.getenv("REWARD_ALIVE", "1.0")))
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=float(os.getenv("REWARD_TERMINATING", "-5.0")))
    # (3) Primary task: end-effector reach target marker position
    end_effector_position = RewTerm(
        func=mdp.end_effector_position_to_marker_l2,
        weight=float(os.getenv("REWARD_END_EFFECTOR_POSITION", "-0.1")),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target_marker"),
            "body_name": "arm_end"  # 末端执行器的链接名称
        },
    )
    # (3.1) Success bonus for reaching target - 提高权重突出成功目标
    target_reached = RewTerm(
        func=mdp.target_reached_bonus,
        weight=float(os.getenv("REWARD_TARGET_REACHED", "20.0")),  # 10.0 -> 20.0
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target_marker"),
            "body_name": "arm_end"
        },
    )
    # (3.2) Distance guidance reward - 降低权重防止过拟合
    distance_guidance = RewTerm(
        func=mdp.distance_guidance_reward,
        weight=float(os.getenv("REWARD_DISTANCE_GUIDANCE", "2.0")),  # 5.0 -> 2.0
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target_marker"),
            "body_name": "arm_end"
        },
    )
    # (3.3) Approach progress reward
    approach_progress = RewTerm(
        func=mdp.approach_progress_reward,
        weight=float(os.getenv("REWARD_APPROACH_PROGRESS", "1.0")),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target_marker"),
            "body_name": "arm_end"
        },
    )
    # (3.4) Convergence monitoring (weight=0, just for monitoring)
    convergence_monitor = RewTerm(
        func=mdp.convergence_monitor,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target_marker"),
            "body_name": "arm_end"
        },
    )
    # (3.5) Termination monitoring (weight=0, just for debugging)
    termination_debug = RewTerm(
        func=mdp.termination_monitor,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # (4) Collision detection and notification
    # collision_detection = RewTerm(
    #     func=mdp.end_effector_target_collision_detection,
    #     weight=0.0,  # 权重为0，仅用于检测和提示，不影响奖励
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "target_cfg": SceneEntityCfg("target_marker"),
    #         "body_name": "arm_end",
    #         "threshold": 0.03  # 3cm碰撞阈值
    #     },
    # )
    # === 反懒惰奖励机制 (轻量化，防止过拟合) ===
    # (5) 关节运动奖励 - 鼓励适度运动，防止停滞
    joint_velocity_reward = RewTerm(
        func=mdp.joint_velocity_reward,
        weight=float(os.getenv("REWARD_JOINT_VELOCITY", "0.1")),  # 0.5 -> 0.1
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # (6) 探索奖励 - 鼓励探索新区域
    exploration_bonus = RewTerm(
        func=mdp.exploration_reward,
        weight=float(os.getenv("REWARD_EXPLORATION", "0.05")),  # 0.3 -> 0.05
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target_marker"),
            "body_name": "arm_end"
        },
    )
    # (7) 反停滞奖励 - 惩罚长时间无改善的行为
    anti_stagnation = RewTerm(
        func=mdp.anti_stagnation_reward,
        weight=float(os.getenv("REWARD_ANTI_STAGNATION", "0.2")),  # 1.0 -> 0.2
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target_marker"), 
            "body_name": "arm_end"
        },
    )
    
    # === 原有的平滑控制奖励（权重降低，防止过度约束）===
    # (8) Shaping tasks: minimize excessive joint velocities
    joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=float(os.getenv("REWARD_JOINT_VEL", "-0.00005")),  # 降低权重，减少对运动的抑制
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-8]"])},
    )
    # (9) Shaping tasks: smooth motion control
    joint_vel_smooth = RewTerm(
        func=mdp.joint_vel_l1,
        weight=float(os.getenv("REWARD_JOINT_VEL_SMOOTH", "-0.0001")),  # 降低权重
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-8]"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Arm joints out of bounds - 关节3,6,7: -270°到0°  
    arm_joints_negative_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[2-7]"]), "bounds": (-3.0 * math.pi, 3.0 * math.pi)},
    )
    # (3) Arm joints out of bounds - 关节2,4,5: 0°到270°
    # arm_joints_positive_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_2","joint_4","joint_5"]), "bounds": (0, 1.5 * math.pi)},
    # )
    # (4) End effector joints out of bounds
    end_effector_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_1","joint_8"]), "bounds": (-math.pi, 3 * math.pi)},
    )


##
# Environment configuration
##


@configclass
class ArmEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings - read num_envs from environment variable
    scene: ArmSceneCfg = ArmSceneCfg(num_envs=int(os.getenv("NUM_ENVS", "2048")), env_spacing=2)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        # Read episode length from environment variable
        self.episode_length_s = float(os.getenv("EPISODE_LENGTH_S", "20"))
        # viewer settings
        self.viewer.eye = (8.0, 5.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation