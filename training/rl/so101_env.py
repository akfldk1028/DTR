#!/usr/bin/env python3
"""so101_env.py — Isaac Lab SO-ARM101 Reach 환경 정의

목적:
    Isaac Lab 기반 SO-ARM101 로봇의 Reach task 강화학습 환경을 정의한다.
    Manager-based environment 패턴을 따르며, 다음 구성요소를 포함한다:

    - SO101_CFG: ArticulationCfg — 로봇 에셋 및 액추에이터 설정
    - SO101ReachSceneCfg: 씬 구성 (로봇, 지면, 목표 마커)
    - SO101ReachObservationsCfg: 관측 공간 (조인트 위치/속도, 목표 위치)
    - SO101ReachActionsCfg: 액션 공간 (6 DOF 조인트 위치 목표)
    - SO101ReachRewardsCfg: 보상 함수 (거리, 액션 패널티)
    - SO101ReachEventCfg: 도메인 랜덤화 이벤트
    - SO101ReachEnvCfg: 환경 설정 통합

    로봇 파라미터는 params/control.yaml, params/physics.yaml에서 참조하며,
    학습 설정은 training/rl/config.yaml에서 로드한다.

사용법:
    # Isaac Lab 환경 등록 후 사용
    import gymnasium as gym
    env = gym.make("Isaac-SO101-Reach-v0", num_envs=1024)

    # 또는 직접 import
    from training.rl.so101_env import SO101ReachEnvCfg
    env_cfg = SO101ReachEnvCfg()

필요 환경:
    - Isaac Sim 5.1.0 + Isaac Lab v2.3.0
    - GPU: NVIDIA RTX 4090 이상 권장
    - conda env: soarm

참고:
    - MuammerBay/isaac_so_arm101: docs/references.md 참조
    - Isaac Lab Manager-based Env: docs/references.md 참조
    - 로봇 제어 파라미터: params/control.yaml
    - 물리 파라미터: params/physics.yaml
    - RL 학습 설정: training/rl/config.yaml

Phase: 6
상태: Isaac Lab RL baseline 환경
"""

from __future__ import annotations

import logging
import math
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# --- 프로젝트 루트 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Isaac Lab imports (런타임에만 사용 가능) ---
# Isaac Sim/Lab이 설치되지 않은 환경에서도 모듈 로드가 가능하도록
# 조건부 임포트를 사용한다.
try:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.actuators import ImplicitActuatorCfg
    from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
    from omni.isaac.lab.envs import ManagerBasedEnvCfg, ManagerBasedRLEnvCfg
    from omni.isaac.lab.managers import (
        ActionTermCfg,
        EventTermCfg,
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        SceneEntityCfg,
        TerminationTermCfg,
    )
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.terrains import TerrainImporterCfg
    from omni.isaac.lab.utils import configclass

    ISAAC_LAB_AVAILABLE = True

except ImportError:
    ISAAC_LAB_AVAILABLE = False
    logger.warning(
        "Isaac Lab을 임포트할 수 없습니다. "
        "Isaac Sim 5.1.0 + Isaac Lab v2.3.0 환경에서 실행하세요."
    )

    # 타입 힌팅과 클래스 정의를 위한 더미 데코레이터/베이스
    def configclass(cls):
        """Isaac Lab configclass 대체 데코레이터."""
        return cls

    class _DummyCfg:
        """Isaac Lab 미설치 환경용 더미 설정 베이스."""
        pass

    class InteractiveSceneCfg(_DummyCfg):
        pass

    class ManagerBasedRLEnvCfg(_DummyCfg):
        pass

    class ArticulationCfg(_DummyCfg):
        pass


# ============================================================================
# 로봇 파라미터 로드
# ============================================================================

def load_yaml_params(filename):
    """params/ 디렉토리에서 YAML 파라미터를 로드한다.

    Args:
        filename: 파라미터 파일명 (예: "control.yaml")

    Returns:
        dict: 파라미터 딕셔너리, 파일이 없으면 None
    """
    import yaml

    params_path = PROJECT_ROOT / "params" / filename
    if not params_path.exists():
        logger.warning("파라미터 파일을 찾을 수 없습니다: %s", params_path)
        return None

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    logger.debug("파라미터 로드 완료: %s", params_path)
    return params


def load_rl_config():
    """training/rl/config.yaml에서 RL 학습 설정을 로드한다.

    Returns:
        dict: RL 설정 딕셔너리, 파일이 없으면 None
    """
    import yaml

    config_path = PROJECT_ROOT / "training" / "rl" / "config.yaml"
    if not config_path.exists():
        logger.warning("RL 설정 파일을 찾을 수 없습니다: %s", config_path)
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug("RL 설정 로드 완료: %s", config_path)
    return config


# ============================================================================
# 로봇 상수 — params/control.yaml 기반
# ============================================================================

# 조인트 이름 (6 DOF: 5 회전 + 1 그리퍼)
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_JOINT_NAMES = ["gripper"]
ALL_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

# 드라이브 게인 — params/control.yaml → drive 참조
ARM_STIFFNESS = 40.0       # N*m/rad — 위치 제어 강성 (Kp)
ARM_DAMPING = 4.0          # N*m*s/rad — 위치 제어 감쇠 (Kd)
GRIPPER_STIFFNESS = 40.0   # N*m/rad — 그리퍼 강성
GRIPPER_DAMPING = 4.0      # N*m*s/rad — 그리퍼 감쇠

# 시뮬레이션 — params/physics.yaml → simulation 참조
SIM_DT = 0.005             # s — 시뮬레이션 시간 간격
DECIMATION = 4             # 시뮬레이션 스텝 / 제어 스텝 비율

# 에피소드 설정 — training/rl/config.yaml → environment 참조
EPISODE_LENGTH = 200       # 에피소드 최대 스텝 수
NUM_ENVS = 1024            # 병렬 환경 수

# 보상 — training/rl/config.yaml → reward 참조
REACHING_WEIGHT = 1.0      # 목표 도달 거리 보상 가중치
ACTION_PENALTY_WEIGHT = 0.01  # 액션 크기 패널티 가중치
SUCCESS_THRESHOLD = 0.02   # m — 목표 도달 판정 거리 임계값

# USD 에셋 경로
USD_PATH = "assets/usd/so101_follower.usd"


# ============================================================================
# ArticulationCfg — SO-ARM101 로봇 에셋 설정
# ============================================================================

if ISAAC_LAB_AVAILABLE:

    SO101_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
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
                joint_names_expr=ARM_JOINT_NAMES,
                stiffness=ARM_STIFFNESS,
                damping=ARM_DAMPING,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=GRIPPER_JOINT_NAMES,
                stiffness=GRIPPER_STIFFNESS,
                damping=GRIPPER_DAMPING,
            ),
        },
    )
    """SO-ARM101 ArticulationCfg.

    6 DOF 로봇 (5 회전 관절 + 1 그리퍼).
    드라이브 게인은 params/control.yaml에서 참조한다:
        - arm: stiffness=40.0, damping=4.0
        - gripper: stiffness=40.0, damping=4.0

    USD 에셋: assets/usd/so101_follower.usd
    참조: MuammerBay/isaac_so_arm101
    """

    # ====================================================================
    # Scene Configuration
    # ====================================================================

    @configclass
    class SO101ReachSceneCfg(InteractiveSceneCfg):
        """SO-ARM101 Reach task 씬 설정.

        씬 구성:
            - robot: SO-ARM101 로봇 (ArticulationCfg)
            - ground: 지면 평면
            - target: 목표 위치 시각화 마커
        """

        # 지면 평면
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        # 조명
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(
                color=(0.9, 0.9, 0.9),
                intensity=500.0,
            ),
        )

        # SO-ARM101 로봇
        robot = SO101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # 목표 위치 마커 (시각화용 소형 구체)
        target = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        )

    # ====================================================================
    # Observation Configuration
    # ====================================================================

    @configclass
    class SO101ReachObservationsCfg:
        """SO-ARM101 Reach task 관측 설정.

        관측 공간 (총 15차원):
            - joint_pos: 조인트 위치 [6]
            - joint_vel: 조인트 속도 [6]
            - target_pos: 목표 위치 [3] (로봇 베이스 기준)

        참조: training/rl/config.yaml → observation
        """

        @configclass
        class PolicyObsCfg(ObservationGroupCfg):
            """정책 네트워크에 입력되는 관측 그룹."""

            # 조인트 위치 (6 DOF)
            joint_pos = ObservationTermCfg(
                func="omni.isaac.lab.envs.mdp.joint_pos_rel",
            )
            # 조인트 속도 (6 DOF)
            joint_vel = ObservationTermCfg(
                func="omni.isaac.lab.envs.mdp.joint_vel_rel",
            )
            # 엔드이펙터와 목표 사이 상대 위치 (3)
            target_pos = ObservationTermCfg(
                func=_compute_target_relative_pos,
            )

        policy: PolicyObsCfg = PolicyObsCfg()

    # ====================================================================
    # Action Configuration
    # ====================================================================

    @configclass
    class SO101ReachActionsCfg:
        """SO-ARM101 Reach task 액션 설정.

        액션 공간 (6차원): 조인트 위치 목표값 [6 DOF]
        타입: 위치 제어 (joint_position)

        참조: training/rl/config.yaml → action
        """

        joint_position = ActionTermCfg(
            asset_name="robot",
            action_type="omni.isaac.lab.envs.mdp.JointPositionAction",
            joint_names=ALL_JOINT_NAMES,
            scale=1.0,
        )

    # ====================================================================
    # Reward Configuration
    # ====================================================================

    @configclass
    class SO101ReachRewardsCfg:
        """SO-ARM101 Reach task 보상 설정.

        보상 항목:
            - reaching: 엔드이펙터-목표 거리 기반 보상 (가중치: 1.0)
            - action_penalty: 액션 크기 패널티 (가중치: -0.01)
            - success_bonus: 목표 도달 시 보너스

        참조: training/rl/config.yaml → reward
        """

        # 목표 도달 거리 보상 (거리가 가까울수록 높은 보상)
        reaching = RewardTermCfg(
            func=_reward_reaching,
            weight=REACHING_WEIGHT,
        )

        # 액션 크기 패널티 (에너지 효율)
        action_penalty = RewardTermCfg(
            func=_reward_action_penalty,
            weight=-ACTION_PENALTY_WEIGHT,
        )

        # 목표 도달 성공 보너스
        success_bonus = RewardTermCfg(
            func=_reward_success_bonus,
            weight=1.0,
        )

    # ====================================================================
    # Event (Domain Randomization) Configuration
    # ====================================================================

    @configclass
    class SO101ReachEventCfg:
        """SO-ARM101 Reach task 도메인 랜덤화 이벤트 설정.

        랜덤화 항목 (spec 기준):
            - 질량: ±20% (params/physics.yaml → mass)
            - 마찰: ±30% (params/physics.yaml → friction)
            - 액추에이터 게인: ±15% (params/control.yaml → drive)
            - 초기 조인트 상태 노이즈

        참조: training/rl/config.yaml → domain_randomization
        """

        # 에피소드 시작 시 로봇 초기 상태 리셋
        reset_robot = EventTermCfg(
            func="omni.isaac.lab.envs.mdp.reset_joints_by_offset",
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": (-0.1, 0.1),
                "velocity_range": (-0.05, 0.05),
            },
        )

        # 에피소드 시작 시 목표 위치 랜덤 배치
        reset_target = EventTermCfg(
            func=_randomize_target_position,
            mode="reset",
        )

        # 질량 랜덤화 (±20%) — 학습 시작 시 적용
        randomize_mass = EventTermCfg(
            func="omni.isaac.lab.envs.mdp.randomize_rigid_body_mass",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )

        # 마찰 랜덤화 (±30%) — 학습 시작 시 적용
        randomize_friction = EventTermCfg(
            func="omni.isaac.lab.envs.mdp.randomize_rigid_body_material",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "static_friction_range": (0.35, 0.65),
                "dynamic_friction_range": (0.28, 0.52),
                "restitution_range": (0.0, 0.0),
            },
        )

        # 액추에이터 게인 랜덤화 (±15%) — 학습 시작 시 적용
        randomize_actuator_gains = EventTermCfg(
            func="omni.isaac.lab.envs.mdp.randomize_actuator_gains",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_distribution_params": (
                    ARM_STIFFNESS * 0.85,
                    ARM_STIFFNESS * 1.15,
                ),
                "damping_distribution_params": (
                    ARM_DAMPING * 0.85,
                    ARM_DAMPING * 1.15,
                ),
            },
        )

    # ====================================================================
    # Termination Configuration
    # ====================================================================

    @configclass
    class SO101ReachTerminationsCfg:
        """SO-ARM101 Reach task 종료 조건 설정.

        종료 조건:
            - time_out: 에피소드 최대 길이 도달
        """

        # 에피소드 최대 길이 도달 시 종료
        time_out = TerminationTermCfg(
            func="omni.isaac.lab.envs.mdp.time_out",
            time_out=True,
        )

    # ====================================================================
    # Environment Configuration (통합)
    # ====================================================================

    @configclass
    class SO101ReachEnvCfg(ManagerBasedRLEnvCfg):
        """SO-ARM101 Reach task 강화학습 환경 설정.

        Isaac Lab Manager-based RL 환경으로,
        SO-ARM101 로봇이 목표 위치에 엔드이펙터를 도달시키는 Reach task를 정의한다.

        구성요소:
            - scene: SO101ReachSceneCfg (로봇, 지면, 목표)
            - observations: SO101ReachObservationsCfg (joint_pos, joint_vel, target_pos)
            - actions: SO101ReachActionsCfg (6 DOF 조인트 위치)
            - rewards: SO101ReachRewardsCfg (거리, 액션 패널티)
            - events: SO101ReachEventCfg (도메인 랜덤화)
            - terminations: SO101ReachTerminationsCfg (시간 초과)

        참조:
            - params/control.yaml: 드라이브 게인, 조인트 리밋
            - params/physics.yaml: 시뮬레이션 dt, 중력
            - training/rl/config.yaml: 학습 하이퍼파라미터
        """

        # 씬 설정
        scene: SO101ReachSceneCfg = SO101ReachSceneCfg(
            num_envs=NUM_ENVS,
            env_spacing=1.5,
        )

        # 관측 설정
        observations: SO101ReachObservationsCfg = SO101ReachObservationsCfg()

        # 액션 설정
        actions: SO101ReachActionsCfg = SO101ReachActionsCfg()

        # 보상 설정
        rewards: SO101ReachRewardsCfg = SO101ReachRewardsCfg()

        # 이벤트 (도메인 랜덤화) 설정
        events: SO101ReachEventCfg = SO101ReachEventCfg()

        # 종료 조건 설정
        terminations: SO101ReachTerminationsCfg = SO101ReachTerminationsCfg()

        # 시뮬레이션 설정
        sim = sim_utils.SimulationCfg(
            dt=SIM_DT,
            render_interval=DECIMATION,
        )

        # 에피소드 설정
        episode_length_s = EPISODE_LENGTH * SIM_DT * DECIMATION

        def __post_init__(self):
            """설정 초기화 후 검증 및 params 교차검증."""
            super().__post_init__()

            # decimation 설정
            self.decimation = DECIMATION

            # params/control.yaml 교차검증 (옵션)
            self._cross_validate_params()

        def _cross_validate_params(self):
            """params/control.yaml, params/physics.yaml과 설정값을 교차검증한다."""
            control_params = load_yaml_params("control.yaml")
            if control_params is None:
                return

            # 드라이브 게인 검증
            drive = control_params.get("drive", {})
            ctrl_stiffness = drive.get("stiffness", {}).get("value")
            ctrl_damping = drive.get("damping", {}).get("value")

            if ctrl_stiffness is not None and ctrl_stiffness != ARM_STIFFNESS:
                logger.warning(
                    "arm_stiffness 불일치 — env: %s, params/control.yaml: %s",
                    ARM_STIFFNESS,
                    ctrl_stiffness,
                )

            if ctrl_damping is not None and ctrl_damping != ARM_DAMPING:
                logger.warning(
                    "arm_damping 불일치 — env: %s, params/control.yaml: %s",
                    ARM_DAMPING,
                    ctrl_damping,
                )

            # 시뮬레이션 dt 검증
            physics_params = load_yaml_params("physics.yaml")
            if physics_params is not None:
                sim_timestep = (
                    physics_params.get("simulation", {})
                    .get("timestep", {})
                    .get("value")
                )
                if sim_timestep is not None and sim_timestep != SIM_DT:
                    logger.warning(
                        "sim_dt 불일치 — env: %s, params/physics.yaml: %s",
                        SIM_DT,
                        sim_timestep,
                    )

else:
    # Isaac Lab 미설치 환경 — 구조 정의만 제공 (타입 힌팅 및 테스트용)

    class SO101ReachEnvCfg:
        """SO-ARM101 Reach task 환경 설정 (Isaac Lab 미설치 환경용 스텁).

        Isaac Lab이 설치된 환경에서 실행하면 전체 기능이 활성화된다.
        이 스텁은 모듈 임포트와 구문 검증만을 위한 것이다.

        구성요소:
            - scene: SO101ReachSceneCfg
            - observations: SO101ReachObservationsCfg
            - actions: SO101ReachActionsCfg
            - rewards: SO101ReachRewardsCfg
            - events: SO101ReachEventCfg
            - terminations: SO101ReachTerminationsCfg

        참조:
            - params/control.yaml: 드라이브 게인, 조인트 리밋
            - params/physics.yaml: 시뮬레이션 dt, 중력
            - training/rl/config.yaml: 학습 하이퍼파라미터
        """

        def __init__(self):
            self.num_envs = NUM_ENVS
            self.episode_length = EPISODE_LENGTH
            self.sim_dt = SIM_DT
            self.decimation = DECIMATION

            logger.info(
                "SO101ReachEnvCfg 스텁 초기화 (Isaac Lab 미설치). "
                "전체 기능은 Isaac Lab 환경에서 사용 가능합니다."
            )


# ============================================================================
# 보상/관측/이벤트 함수
# ============================================================================
# Isaac Lab의 manager-based env는 보상/관측 함수를 별도 정의하여 참조한다.
# 런타임에 Isaac Lab에서 호출하는 함수들이다.

def _compute_target_relative_pos(env):
    """엔드이펙터와 목표 사이의 상대 위치를 계산한다.

    Args:
        env: Isaac Lab ManagerBasedRLEnv 인스턴스

    Returns:
        torch.Tensor: 상대 위치 벡터 [num_envs, 3]
    """
    # 로봇 엔드이펙터 위치 (마지막 링크의 월드 좌표)
    robot = env.scene["robot"]
    ee_pos = robot.data.body_pos_w[:, -1, :3]

    # 목표 위치
    target = env.scene["target"]
    target_pos = target.data.root_pos_w[:, :3]

    # 상대 위치 (로봇 베이스 기준)
    robot_base_pos = robot.data.root_pos_w[:, :3]
    relative_pos = target_pos - ee_pos

    return relative_pos


def _reward_reaching(env):
    """목표 도달 거리 기반 보상을 계산한다.

    엔드이펙터와 목표 사이의 유클리드 거리가 가까울수록 높은 보상.
    보상 = 1 - tanh(distance * 5)

    Args:
        env: Isaac Lab ManagerBasedRLEnv 인스턴스

    Returns:
        torch.Tensor: 보상 값 [num_envs]
    """
    import torch

    relative_pos = _compute_target_relative_pos(env)
    distance = torch.norm(relative_pos, dim=-1)

    # tanh 기반 보상: 거리 0이면 보상 1, 거리가 멀면 보상 0에 수렴
    reward = 1.0 - torch.tanh(distance * 5.0)
    return reward


def _reward_action_penalty(env):
    """액션 크기 패널티를 계산한다.

    큰 액션에 패널티를 부여하여 에너지 효율적인 동작을 유도한다.
    패널티 = sum(action^2)

    Args:
        env: Isaac Lab ManagerBasedRLEnv 인스턴스

    Returns:
        torch.Tensor: 패널티 값 [num_envs]
    """
    import torch

    actions = env.action_manager.action
    penalty = torch.sum(actions ** 2, dim=-1)
    return penalty


def _reward_success_bonus(env):
    """목표 도달 성공 보너스를 계산한다.

    엔드이펙터가 목표 위치에 충분히 가까우면 (< success_threshold) 보너스 지급.

    Args:
        env: Isaac Lab ManagerBasedRLEnv 인스턴스

    Returns:
        torch.Tensor: 성공 보너스 [num_envs] (0 또는 1)
    """
    import torch

    relative_pos = _compute_target_relative_pos(env)
    distance = torch.norm(relative_pos, dim=-1)

    # 목표 도달 판정 — training/rl/config.yaml → reward.success_threshold
    success = (distance < SUCCESS_THRESHOLD).float()
    return success


def _randomize_target_position(env, env_ids):
    """목표 위치를 로봇 작업 공간 내에서 랜덤하게 배치한다.

    SO-ARM101의 작업 공간을 고려하여 도달 가능한 범위 내에서
    목표 위치를 균일 분포로 샘플링한다.

    Args:
        env: Isaac Lab ManagerBasedRLEnv 인스턴스
        env_ids: 리셋할 환경 인덱스
    """
    import torch

    num_resets = len(env_ids)
    device = env.device

    # SO-ARM101 작업 공간 범위 (m)
    # 로봇 베이스 기준 도달 가능 영역 (대략적 추정)
    target_pos = torch.zeros(num_resets, 3, device=device)
    target_pos[:, 0] = torch.uniform_(
        torch.empty(num_resets, device=device), -0.15, 0.15
    )  # x: ±15cm
    target_pos[:, 1] = torch.uniform_(
        torch.empty(num_resets, device=device), -0.15, 0.15
    )  # y: ±15cm
    target_pos[:, 2] = torch.uniform_(
        torch.empty(num_resets, device=device), 0.05, 0.25
    )  # z: 5~25cm (지면 위)

    # 로봇 베이스 위치 기준 오프셋 적용
    robot_base_pos = env.scene["robot"].data.root_pos_w[env_ids, :3]
    target_pos += robot_base_pos

    # 목표 위치 업데이트
    target = env.scene["target"]
    target.write_root_pose_to_sim(
        root_pos=target_pos,
        env_ids=env_ids,
    )
