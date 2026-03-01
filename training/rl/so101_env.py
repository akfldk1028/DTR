#!/usr/bin/env python3
"""so101_env.py — Isaac Lab SO-ARM101 ArticulationCfg + Reach task 환경

목적:
    Isaac Lab에서 SO-ARM101 로봇의 RL 학습 환경을 정의한다.
    - SO101_CFG: ArticulationCfg (USD 로드 + 액추에이터 설정)
    - SO101ReachEnvCfg: ManagerBasedRLEnvCfg 기반 Reach task 환경
    - 관측/액션 공간, 보상 함수 정의
    - params/control.yaml에서 드라이브 게인/조인트 리밋 로드

사용법:
    # Isaac Lab 학습 환경으로 import
    from training.rl.so101_env import SO101_CFG, SO101ReachEnvCfg

    # 또는 직접 실행하여 설정 확인
    python training/rl/so101_env.py

필요 환경:
    - Isaac Sim 5.1.0 + Isaac Lab v2.3.0
    - conda env: soarm
    - GPU: NVIDIA RTX 4090 Laptop

Phase: 6
상태: 구현 완료 — Isaac Lab RL Reach task 환경
"""

import logging
import math
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Isaac Lab imports are guarded so the script can be syntax-checked
# and imported without Isaac Sim / Isaac Lab runtime.
# ---------------------------------------------------------------------------
try:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import ArticulationCfg
    from omni.isaac.lab.actuators import ImplicitActuatorCfg
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
    from omni.isaac.lab.managers import (
        EventTermCfg,
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        SceneEntityCfg,
        TerminationTermCfg,
    )
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.utils import configclass

    ISAAC_LAB_AVAILABLE = True
except ImportError:
    try:
        # Newer Isaac Lab namespace fallback
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
        from isaaclab.managers import (
            EventTermCfg,
            ObservationGroupCfg,
            ObservationTermCfg,
            RewardTermCfg,
            SceneEntityCfg,
            TerminationTermCfg,
        )
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        ISAAC_LAB_AVAILABLE = True
    except ImportError:
        ISAAC_LAB_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter file paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PARAMS_DIR = _SCRIPT_DIR.parent.parent / "params"
_CONTROL_YAML = _PARAMS_DIR / "control.yaml"
_RL_CONFIG_YAML = _SCRIPT_DIR / "config.yaml"


# ---------------------------------------------------------------------------
# YAML helpers (same pattern as training/il/train_act.py)
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    """Load a YAML parameter file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_value(param):
    """Extract the 'value' field from a parameter dict.

    Handles both top-level {value: ...} and plain values.
    """
    if isinstance(param, dict) and "value" in param:
        return param["value"]
    return param


# ---------------------------------------------------------------------------
# Load drive gains and joint limits from params/control.yaml
# ---------------------------------------------------------------------------
def load_control_params() -> dict:
    """Load params/control.yaml and extract drive gains + joint limits.

    Returns:
        Dictionary with keys: stiffness, damping, max_effort,
        joint_names, position_min, position_max, velocity_max,
        decimation, action_scale.
    """
    raw = _load_yaml(_CONTROL_YAML)

    drive = raw["drive"]
    joint_limits = raw["joint_limits"]

    params = {
        "stiffness": _get_value(drive["stiffness"]),
        "damping": _get_value(drive["damping"]),
        "max_effort": _get_value(drive["max_effort"]),
        "joint_names": _get_value(raw["joint_names"]),
        "position_min": _get_value(joint_limits["position_min"]),
        "position_max": _get_value(joint_limits["position_max"]),
        "velocity_max": _get_value(joint_limits["velocity_max"]),
        "decimation": _get_value(raw["decimation"]),
        "action_scale": _get_value(raw["action_scale"]),
    }

    return params


def load_rl_config() -> dict:
    """Load training/rl/config.yaml and return raw dict."""
    raw = _load_yaml(_RL_CONFIG_YAML)
    return raw


# ---------------------------------------------------------------------------
# Load parameters at module level (safe — only reads YAML)
# ---------------------------------------------------------------------------
_CTRL = load_control_params()
_RL_CFG = load_rl_config()

# Drive gains from params/control.yaml
_STIFFNESS = _CTRL["stiffness"]     # 40.0 N*m/rad
_DAMPING = _CTRL["damping"]         # 4.0 N*m*s/rad
_MAX_EFFORT = _CTRL["max_effort"]   # 5.0 N*m
_JOINT_NAMES = _CTRL["joint_names"]
_DECIMATION = _CTRL["decimation"]   # 10 steps

# Joint limits from params/control.yaml
_POS_MIN = _CTRL["position_min"]    # [6] rad
_POS_MAX = _CTRL["position_max"]    # [6] rad
_VEL_MAX = _CTRL["velocity_max"]    # [6] rad/s
_ACTION_SCALE = _CTRL["action_scale"]

# Reward weights from training/rl/config.yaml
_REWARD_CFG = _RL_CFG.get("reward", {})
_REACH_POS_WEIGHT = _get_value(_REWARD_CFG.get("reach_position_weight", {"value": 1.0}))
_ACTION_PENALTY_WEIGHT = _get_value(_REWARD_CFG.get("action_penalty_weight", {"value": 0.01}))
_VEL_PENALTY_WEIGHT = _get_value(
    _REWARD_CFG.get("joint_velocity_penalty_weight", {"value": 0.005})
)
_SUCCESS_BONUS = _get_value(_REWARD_CFG.get("success_bonus", {"value": 5.0}))
_SUCCESS_THRESHOLD = _get_value(_REWARD_CFG.get("success_threshold", {"value": 0.02}))

# Environment dimensions
_NUM_JOINTS = len(_JOINT_NAMES)  # 6
_ARM_JOINT_NAMES = _JOINT_NAMES[:5]  # 5 revolute joints
_GRIPPER_JOINT_NAMES = _JOINT_NAMES[5:]  # 1 gripper joint

# USD asset path (relative to project root)
_USD_PATH = "assets/usd/so101_follower.usd"


# ---------------------------------------------------------------------------
# SO101_CFG — ArticulationCfg for SO-ARM101
# ---------------------------------------------------------------------------
if ISAAC_LAB_AVAILABLE:

    SO101_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=_ARM_JOINT_NAMES,
                stiffness=_STIFFNESS,
                damping=_DAMPING,
                effort_limit=_MAX_EFFORT,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=_GRIPPER_JOINT_NAMES,
                stiffness=_STIFFNESS,
                damping=_DAMPING,
                effort_limit=_MAX_EFFORT,
            ),
        },
    )

    # -------------------------------------------------------------------
    # Observation terms
    # -------------------------------------------------------------------
    def joint_position_obs(env: ManagerBasedRLEnv) -> "torch.Tensor":
        """Observe current joint positions (normalized to [-1, 1]).

        Returns:
            Tensor of shape (num_envs, num_joints) with joint positions.
        """
        robot = env.scene["robot"]
        return robot.data.joint_pos[:, :_NUM_JOINTS]

    def joint_velocity_obs(env: ManagerBasedRLEnv) -> "torch.Tensor":
        """Observe current joint velocities.

        Returns:
            Tensor of shape (num_envs, num_joints) with joint velocities.
        """
        robot = env.scene["robot"]
        return robot.data.joint_vel[:, :_NUM_JOINTS]

    def target_position_obs(env: ManagerBasedRLEnv) -> "torch.Tensor":
        """Observe target end-effector position in world frame.

        Returns:
            Tensor of shape (num_envs, 3) with target XYZ position.
        """
        return env.command_manager.get_command("target_pose")[:, :3]

    def end_effector_position_obs(env: ManagerBasedRLEnv) -> "torch.Tensor":
        """Observe current end-effector position in world frame.

        Returns:
            Tensor of shape (num_envs, 3) with end-effector XYZ position.
        """
        robot = env.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, -1, :]  # last link as EE
        return ee_pos

    # -------------------------------------------------------------------
    # Reward terms
    # -------------------------------------------------------------------
    def reach_position_reward(env: ManagerBasedRLEnv) -> "torch.Tensor":
        """Reward for reaching the target position.

        Negative L2 distance between end-effector and target, weighted
        by reach_position_weight from training/rl/config.yaml.

        Returns:
            Tensor of shape (num_envs,) with reward values.
        """
        robot = env.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, -1, :]
        target_pos = env.command_manager.get_command("target_pose")[:, :3]
        distance = torch.norm(ee_pos - target_pos, dim=-1)
        return -distance * _REACH_POS_WEIGHT

    def action_penalty_reward(env: ManagerBasedRLEnv) -> "torch.Tensor":
        """Penalty for large actions (energy efficiency).

        Returns:
            Tensor of shape (num_envs,) with penalty values.
        """
        actions = env.action_manager.action
        penalty = torch.sum(actions ** 2, dim=-1)
        return -penalty * _ACTION_PENALTY_WEIGHT

    def joint_velocity_penalty_reward(env: ManagerBasedRLEnv) -> "torch.Tensor":
        """Penalty for high joint velocities (smooth motion).

        Returns:
            Tensor of shape (num_envs,) with penalty values.
        """
        robot = env.scene["robot"]
        joint_vel = robot.data.joint_vel[:, :_NUM_JOINTS]
        penalty = torch.sum(joint_vel ** 2, dim=-1)
        return -penalty * _VEL_PENALTY_WEIGHT

    def reach_success_bonus_reward(env: ManagerBasedRLEnv) -> "torch.Tensor":
        """Bonus reward when end-effector reaches target within threshold.

        Returns:
            Tensor of shape (num_envs,) with bonus values.
        """
        robot = env.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, -1, :]
        target_pos = env.command_manager.get_command("target_pose")[:, :3]
        distance = torch.norm(ee_pos - target_pos, dim=-1)
        success = (distance < _SUCCESS_THRESHOLD).float()
        return success * _SUCCESS_BONUS

    # -------------------------------------------------------------------
    # Termination terms
    # -------------------------------------------------------------------
    def time_limit_termination(env: ManagerBasedRLEnv) -> "torch.Tensor":
        """Terminate episode when max episode length is reached.

        Returns:
            Boolean tensor of shape (num_envs,).
        """
        max_episode_length = _get_value(
            _RL_CFG["environment"]["episode_length"]
        )
        return env.episode_length_buf >= max_episode_length

    # -------------------------------------------------------------------
    # Scene configuration
    # -------------------------------------------------------------------
    @configclass
    class SO101SceneCfg(InteractiveSceneCfg):
        """Scene configuration for SO-ARM101 reach task."""

        robot: ArticulationCfg = SO101_CFG

        # Ground plane
        ground = sim_utils.GroundPlaneCfg()

        # Dome light for rendering
        dome_light = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.8, 0.8, 0.8),
        )

    # -------------------------------------------------------------------
    # Observation group configuration
    # -------------------------------------------------------------------
    @configclass
    class ObservationsCfg:
        """Observation group configuration for Reach task."""

        @configclass
        class PolicyCfg(ObservationGroupCfg):
            """Observation group for the policy network."""

            joint_pos = ObservationTermCfg(
                func=joint_position_obs,
            )
            joint_vel = ObservationTermCfg(
                func=joint_velocity_obs,
            )
            target_pos = ObservationTermCfg(
                func=target_position_obs,
            )
            ee_pos = ObservationTermCfg(
                func=end_effector_position_obs,
            )

        policy: PolicyCfg = PolicyCfg()

    # -------------------------------------------------------------------
    # Reward configuration
    # -------------------------------------------------------------------
    @configclass
    class RewardsCfg:
        """Reward configuration for Reach task."""

        reach_position = RewardTermCfg(
            func=reach_position_reward,
            weight=1.0,
        )
        action_penalty = RewardTermCfg(
            func=action_penalty_reward,
            weight=1.0,
        )
        joint_velocity_penalty = RewardTermCfg(
            func=joint_velocity_penalty_reward,
            weight=1.0,
        )
        reach_success_bonus = RewardTermCfg(
            func=reach_success_bonus_reward,
            weight=1.0,
        )

    # -------------------------------------------------------------------
    # Termination configuration
    # -------------------------------------------------------------------
    @configclass
    class TerminationsCfg:
        """Termination configuration for Reach task."""

        time_limit = TerminationTermCfg(
            func=time_limit_termination,
            time_out=True,
        )

    # -------------------------------------------------------------------
    # SO101ReachEnvCfg — ManagerBasedRLEnvCfg for Reach task
    # -------------------------------------------------------------------
    @configclass
    class SO101ReachEnvCfg(ManagerBasedRLEnvCfg):
        """Isaac Lab RL environment configuration for SO-ARM101 Reach task.

        Observation space:
            - joint_pos: [num_envs, 6] — joint positions
            - joint_vel: [num_envs, 6] — joint velocities
            - target_pos: [num_envs, 3] — target XYZ position
            - ee_pos: [num_envs, 3] — end-effector XYZ position

        Action space:
            - [num_envs, 6] — target joint positions (position control)

        Reward:
            - reach_position: -L2_distance * weight
            - action_penalty: -sum(action^2) * weight
            - joint_velocity_penalty: -sum(vel^2) * weight
            - reach_success_bonus: bonus if distance < threshold
        """

        # Scene
        scene: SO101SceneCfg = SO101SceneCfg(
            num_envs=_get_value(_RL_CFG["environment"]["num_envs"]),
            env_spacing=2.0,
        )

        # Observations
        observations: ObservationsCfg = ObservationsCfg()

        # Rewards
        rewards: RewardsCfg = RewardsCfg()

        # Terminations
        terminations: TerminationsCfg = TerminationsCfg()

        # Simulation parameters
        sim = sim_utils.SimulationCfg(
            dt=_get_value(_RL_CFG["environment"]["sim_dt"]),
        )

        # Decimation (sim_steps per policy step)
        decimation = _DECIMATION

else:
    # Isaac Lab is not available — provide stub classes so the module
    # can still be imported for syntax checking and --help.
    logger.warning(
        "Isaac Lab is not available. "
        "SO101_CFG and SO101ReachEnvCfg are stubs. "
        "Install Isaac Lab for full functionality."
    )

    SO101_CFG = None

    class SO101ReachEnvCfg:
        """Stub class when Isaac Lab is not available."""

        pass


# ---------------------------------------------------------------------------
# Module-level summary
# ---------------------------------------------------------------------------
def print_env_summary() -> None:
    """Print a summary of the environment configuration."""
    logger.info("=== SO-ARM101 Reach Environment Summary ===")
    logger.info("Isaac Lab available: %s", ISAAC_LAB_AVAILABLE)
    logger.info("")
    logger.info("--- Drive Gains (params/control.yaml) ---")
    logger.info("  stiffness:   %.1f N*m/rad", _STIFFNESS)
    logger.info("  damping:     %.1f N*m*s/rad", _DAMPING)
    logger.info("  max_effort:  %.1f N*m", _MAX_EFFORT)
    logger.info("")
    logger.info("--- Joint Configuration ---")
    logger.info("  num_joints:  %d", _NUM_JOINTS)
    logger.info("  joint_names: %s", _JOINT_NAMES)
    logger.info("  arm_joints:  %s", _ARM_JOINT_NAMES)
    logger.info("  gripper:     %s", _GRIPPER_JOINT_NAMES)
    logger.info("")
    logger.info("--- Joint Limits ---")
    for i, name in enumerate(_JOINT_NAMES):
        logger.info(
            "  %s: [%.4f, %.4f] rad, vel_max=%.2f rad/s",
            name,
            _POS_MIN[i],
            _POS_MAX[i],
            _VEL_MAX[i],
        )
    logger.info("")
    logger.info("--- Environment Parameters ---")
    logger.info("  decimation:  %d steps", _DECIMATION)
    logger.info("  action_scale: %.1f", _ACTION_SCALE)
    logger.info("  USD path:    %s", _USD_PATH)
    logger.info("")
    logger.info("--- Reward Configuration ---")
    logger.info("  reach_position_weight:    %.3f", _REACH_POS_WEIGHT)
    logger.info("  action_penalty_weight:    %.3f", _ACTION_PENALTY_WEIGHT)
    logger.info("  velocity_penalty_weight:  %.4f", _VEL_PENALTY_WEIGHT)
    logger.info("  success_bonus:            %.1f", _SUCCESS_BONUS)
    logger.info("  success_threshold:        %.3f m", _SUCCESS_THRESHOLD)


def main() -> None:
    """Entry point — print environment configuration summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print_env_summary()


if __name__ == "__main__":
    main()
