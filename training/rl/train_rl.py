#!/usr/bin/env python3
"""train_rl.py — skrl PPO 강화학습 래퍼 스크립트

목적:
    skrl 프레임워크를 사용한 PPO 강화학습을 프로젝트 컨벤션에 맞게 래핑한다.
    training/rl/config.yaml에서 하이퍼파라미터를 로드하고,
    training/rl/so101_env.py의 Isaac Lab 환경에서 PPO 에이전트를 학습한다.

    - config.yaml 기반 하이퍼파라미터 로드
    - params/control.yaml 교차검증 (드라이브 게인, 조인트 리밋)
    - Isaac Lab 환경 초기화 (SO101ReachEnvCfg)
    - skrl PPO 에이전트 구성 (공유 MLP actor-critic)
    - 학습 실행 및 체크포인트 저장 (training/rl/checkpoints/)
    - TensorBoard 메트릭 로깅

사용법:
    python training/rl/train_rl.py --task Isaac-SO101-Reach-v0 --num-envs 1024 --headless
    python training/rl/train_rl.py --config training/rl/config.yaml --max-iterations 2000
    python training/rl/train_rl.py --config training/rl/config.yaml --dry-run
    python training/rl/train_rl.py --resume training/rl/checkpoints/best_agent.pt

필요 환경:
    - Isaac Sim 5.1.0 + Isaac Lab v2.3.0
    - skrl (강화학습 프레임워크)
    - PyTorch (GPU 필수)
    - conda env: soarm

참고:
    - skrl 문서: docs/references.md 참조
    - Isaac Lab RL 환경: training/rl/so101_env.py
    - RL 학습 설정: training/rl/config.yaml
    - 로봇 제어 파라미터: params/control.yaml
    - 물리 파라미터: params/physics.yaml

Phase: 6
상태: Isaac Lab RL baseline 학습
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# --- 프로젝트 루트 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="skrl PPO 강화학습 — SO-ARM101 Reach task"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Isaac Lab 태스크 이름 (미지정 시 config에서 읽음)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="병렬 환경 수 (미지정 시 config에서 읽음)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="렌더링 없이 학습 실행",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/rl/config.yaml",
        help="학습 설정 파일 경로 (default: training/rl/config.yaml)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="최대 학습 반복 수 (미지정 시 config에서 읽음)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="이전 체크포인트 경로 (재학습 시)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="설정 검증만 수행하고 학습은 실행하지 않음",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="랜덤 시드 (미지정 시 config에서 읽음)",
    )
    return parser.parse_args()


def load_config(config_path):
    """YAML 설정 파일을 로드한다.

    Args:
        config_path: config.yaml 파일 경로

    Returns:
        dict: 파싱된 설정 딕셔너리

    Raises:
        FileNotFoundError: 설정 파일이 없을 경우
        ValueError: 필수 키가 누락된 경우
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_keys = ["environment", "training", "robot", "observation", "action", "reward"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"설정 파일에 필수 키가 누락되었습니다: {missing}")

    logger.info("설정 로드 완료: %s", config_path)
    return config


def load_control_params(params_path=None):
    """params/control.yaml에서 로봇 제어 파라미터를 로드한다.

    Args:
        params_path: control.yaml 경로 (미지정 시 프로젝트 루트 기준)

    Returns:
        dict: 제어 파라미터 딕셔너리, 파일이 없으면 None
    """
    import yaml

    if params_path is None:
        params_path = PROJECT_ROOT / "params" / "control.yaml"
    else:
        params_path = Path(params_path)

    if not params_path.exists():
        logger.warning("제어 파라미터 파일을 찾을 수 없습니다: %s", params_path)
        return None

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    logger.info("제어 파라미터 로드 완료: %s", params_path)
    return params


def validate_config(config, control_params=None):
    """설정 값의 유효성을 검증한다.

    config.yaml의 로봇 설정과 params/control.yaml의 값을 교차검증한다.

    Args:
        config: config.yaml에서 로드한 설정
        control_params: params/control.yaml에서 로드한 파라미터 (None이면 스킵)

    Raises:
        ValueError: 설정 값이 유효하지 않을 경우
    """
    env_cfg = config["environment"]
    training = config["training"]
    robot = config["robot"]

    # --- 환경 설정 검증 ---
    if env_cfg.get("num_envs", 0) <= 0:
        raise ValueError("num_envs는 양의 정수여야 합니다")

    if env_cfg.get("episode_length", 0) <= 0:
        raise ValueError("episode_length는 양의 정수여야 합니다")

    if env_cfg.get("sim_dt", 0) <= 0:
        raise ValueError("sim_dt는 양수여야 합니다")

    # --- 학습 설정 검증 ---
    if training.get("algorithm") != "PPO":
        raise ValueError(
            f"지원하지 않는 알고리즘: {training.get('algorithm')} (PPO만 지원)"
        )

    if training.get("learning_rate", 0) <= 0:
        raise ValueError("learning_rate는 양수여야 합니다")

    if training.get("max_iterations", 0) <= 0:
        raise ValueError("max_iterations는 양의 정수여야 합니다")

    if training.get("rollout_steps", 0) <= 0:
        raise ValueError("rollout_steps는 양의 정수여야 합니다")

    # --- 로봇 설정 검증 ---
    if robot.get("dof") != 6:
        raise ValueError(f"SO-ARM101은 6 DOF여야 합니다 (현재: {robot.get('dof')})")

    if len(robot.get("joint_names", [])) != 6:
        raise ValueError("joint_names는 6개여야 합니다")

    # --- 네트워크 아키텍처 검증 ---
    network = training.get("network", {})
    hidden_layers = network.get("hidden_layers", [])
    if not hidden_layers:
        raise ValueError("network.hidden_layers가 비어있습니다")

    # --- params/control.yaml 교차검증 ---
    if control_params is not None:
        _cross_validate_params(config, control_params)

    logger.info("설정 검증 완료")


def _cross_validate_params(config, control_params):
    """config.yaml과 params/control.yaml의 교차검증을 수행한다.

    Args:
        config: config.yaml 설정
        control_params: params/control.yaml 파라미터
    """
    robot = config["robot"]

    # 드라이브 게인 비교
    drive = control_params.get("drive", {})
    ctrl_stiffness = drive.get("stiffness", {}).get("value")
    ctrl_damping = drive.get("damping", {}).get("value")

    if ctrl_stiffness is not None and robot.get("arm_stiffness") != ctrl_stiffness:
        logger.warning(
            "arm_stiffness 불일치 — config: %s, params/control.yaml: %s",
            robot.get("arm_stiffness"),
            ctrl_stiffness,
        )

    if ctrl_damping is not None and robot.get("arm_damping") != ctrl_damping:
        logger.warning(
            "arm_damping 불일치 — config: %s, params/control.yaml: %s",
            robot.get("arm_damping"),
            ctrl_damping,
        )

    # 조인트 리밋 비교
    joint_limits = control_params.get("joint_limits", {})
    pos_min = joint_limits.get("position_min", {}).get("value")
    pos_max = joint_limits.get("position_max", {}).get("value")

    if pos_min is not None and robot.get("joint_position_min") != pos_min:
        logger.warning(
            "joint_position_min 불일치 — config: %s, params/control.yaml: %s",
            robot.get("joint_position_min"),
            pos_min,
        )

    if pos_max is not None and robot.get("joint_position_max") != pos_max:
        logger.warning(
            "joint_position_max 불일치 — config: %s, params/control.yaml: %s",
            robot.get("joint_position_max"),
            pos_max,
        )

    # 액션 스케일 비교
    action_scale = control_params.get("action_scale", {}).get("value")
    if action_scale is not None and robot.get("action_scale") != action_scale:
        logger.warning(
            "action_scale 불일치 — config: %s, params/control.yaml: %s",
            robot.get("action_scale"),
            action_scale,
        )

    # 제어 주파수 비교
    ctrl_freq = control_params.get("control_frequency", {}).get("value")
    if ctrl_freq is not None and robot.get("control_frequency") != ctrl_freq:
        logger.warning(
            "control_frequency 불일치 — config: %s, params/control.yaml: %s",
            robot.get("control_frequency"),
            ctrl_freq,
        )


def setup_environment(config, num_envs=None, headless=False, task=None):
    """Isaac Lab 강화학습 환경을 초기화한다.

    SO101ReachEnvCfg를 사용하여 Isaac Lab 환경을 생성한다.
    Isaac Lab이 설치되지 않은 경우 안내 메시지를 출력한다.

    Args:
        config: 학습 설정 딕셔너리
        num_envs: 병렬 환경 수 (None이면 config에서 읽음)
        headless: 렌더링 없이 실행 여부
        task: Isaac Lab 태스크 이름 (None이면 config에서 읽음)

    Returns:
        환경 인스턴스 (Isaac Lab ManagerBasedRLEnv)

    Raises:
        ImportError: Isaac Lab이 설치되지 않은 경우
    """
    env_cfg = config["environment"]

    if num_envs is None:
        num_envs = env_cfg.get("num_envs", 1024)

    if task is None:
        task = env_cfg.get("task_name", "Isaac-SO101-Reach-v0")

    try:
        import gymnasium as gym
        from training.rl.so101_env import SO101ReachEnvCfg

        # Isaac Lab 환경 설정
        env_config = SO101ReachEnvCfg()
        env_config.scene.num_envs = num_envs

        # 환경 생성
        env = gym.make(
            task,
            cfg=env_config,
            render_mode="rgb_array" if not headless else None,
        )

        logger.info(
            "Isaac Lab 환경 초기화 완료: %s (envs: %d, headless: %s)",
            task,
            num_envs,
            headless,
        )
        return env

    except ImportError as e:
        logger.error(
            "Isaac Lab 환경을 초기화할 수 없습니다: %s\n"
            "Isaac Sim 5.1.0 + Isaac Lab v2.3.0을 설치하세요.\n"
            "설치 가이드: docs/references.md 참조",
            e,
        )
        raise


def setup_agent(config, env, resume_from=None):
    """skrl PPO 에이전트를 구성한다.

    config.yaml의 학습 설정에 따라 PPO 에이전트를 생성한다.
    공유 MLP actor-critic 네트워크를 사용한다.

    Args:
        config: 학습 설정 딕셔너리
        env: Isaac Lab 환경 인스턴스
        resume_from: 이전 체크포인트 경로 (재학습 시)

    Returns:
        tuple: (agent, trainer) — skrl PPO 에이전트와 트레이너

    Raises:
        ImportError: skrl이 설치되지 않은 경우
    """
    training = config["training"]
    network_cfg = training.get("network", {})
    obs_cfg = config["observation"]
    action_cfg = config["action"]

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.error(
            "PyTorch가 설치되지 않았습니다. "
            "설치: pip install torch"
        )
        raise

    try:
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
        from skrl.memories.torch import RandomMemory
        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        from skrl.trainers.torch import SequentialTrainer
        from skrl.utils import set_seed
    except ImportError:
        logger.error(
            "skrl이 설치되지 않았습니다. "
            "설치: pip install skrl\n"
            "자세한 내용: docs/references.md 참조"
        )
        raise

    # --- 재현성 설정 ---
    seed = training.get("seed", 42)
    set_seed(seed)
    logger.info("랜덤 시드: %d", seed)

    # --- GPU 확인 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("GPU를 사용할 수 없습니다. CPU로 학습합니다 (매우 느림).")
    else:
        logger.info("GPU 사용: %s", torch.cuda.get_device_name(0))

    # --- 네트워크 아키텍처 ---
    obs_dim = obs_cfg.get("total_dim", 15)
    action_dim = action_cfg.get("dim", 6)
    hidden_layers = network_cfg.get("hidden_layers", [256, 128, 64])

    # 활성화 함수 매핑
    activation_map = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "selu": nn.SELU,
    }
    activation_name = network_cfg.get("activation", "elu")
    activation_cls = activation_map.get(activation_name, nn.ELU)

    # 공유 MLP Actor-Critic 모델 정의
    class SharedActorCritic(GaussianMixin, DeterministicMixin, Model):
        """공유 MLP Actor-Critic 네트워크.

        Actor (Gaussian): 연속 액션 분포를 출력한다.
        Critic (Deterministic): 상태 가치를 출력한다.

        네트워크 아키텍처:
            - 공유 히든 레이어: config.yaml → training.network.hidden_layers
            - 활성화 함수: config.yaml → training.network.activation
            - 초기화: orthogonal
        """

        def __init__(self, observation_space, action_space, device, clip_actions=False,
                     clip_log_std=True, min_log_std=-20, max_log_std=2,
                     reduction="sum", role=""):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(
                self,
                clip_actions=clip_actions,
                clip_log_std=clip_log_std,
                min_log_std=min_log_std,
                max_log_std=max_log_std,
                reduction=reduction,
                role=role,
            )
            DeterministicMixin.__init__(self, clip_actions=clip_actions, role=role)

            # 공유 백본 구성
            layers = []
            in_dim = obs_dim
            for h_dim in hidden_layers:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(activation_cls())
                in_dim = h_dim

            self.backbone = nn.Sequential(*layers)

            # Actor 헤드 (Gaussian mean)
            self.actor_head = nn.Linear(in_dim, action_dim)
            # Actor log_std (학습 가능한 파라미터)
            self.log_std_parameter = nn.Parameter(
                torch.zeros(action_dim)
            )

            # Critic 헤드 (상태 가치)
            self.critic_head = nn.Linear(in_dim, 1)

            # 가중치 초기화
            if network_cfg.get("init_type") == "orthogonal":
                self._init_orthogonal()

        def _init_orthogonal(self):
            """Orthogonal 가중치 초기화."""
            for module in self.backbone:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    nn.init.constant_(module.bias, 0.0)

            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.constant_(self.actor_head.bias, 0.0)

            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
            nn.init.constant_(self.critic_head.bias, 0.0)

        def compute(self, inputs, role=""):
            """순전파 계산.

            Args:
                inputs: 관측 텐서 딕셔너리
                role: 모델 역할 ("policy" 또는 "value")

            Returns:
                Actor 역할: (mean_actions, log_std, {})
                Critic 역할: (value, {})
            """
            features = self.backbone(inputs["states"])

            if role == "policy":
                mean = self.actor_head(features)
                log_std = self.log_std_parameter.expand_as(mean)
                return mean, log_std, {}
            elif role == "value":
                value = self.critic_head(features)
                return value, {}

            # 기본값 (policy)
            mean = self.actor_head(features)
            log_std = self.log_std_parameter.expand_as(mean)
            return mean, log_std, {}

    # --- 모델 인스턴스 생성 ---
    models = {}
    models["policy"] = SharedActorCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        role="policy",
    )
    models["value"] = models["policy"]  # 공유 네트워크

    logger.info(
        "네트워크 아키텍처: 공유 MLP %s, 활성화: %s",
        hidden_layers,
        activation_name,
    )

    # --- 메모리 (경험 버퍼) ---
    rollout_steps = training.get("rollout_steps", 24)
    memory = RandomMemory(
        memory_size=rollout_steps,
        num_envs=env.num_envs,
        device=device,
    )

    # --- PPO 에이전트 설정 ---
    agent_cfg = PPO_DEFAULT_CONFIG.copy()

    # PPO 하이퍼파라미터 — config.yaml → training
    agent_cfg["learning_rate"] = training.get("learning_rate", 3e-4)
    agent_cfg["learning_epochs"] = training.get("num_epochs", 5)
    agent_cfg["mini_batches"] = training.get("num_mini_batches", 4)
    agent_cfg["discount_factor"] = training.get("gamma", 0.99)
    agent_cfg["lambda"] = training.get("lambda_gae", 0.95)
    agent_cfg["ratio_clip"] = training.get("clip_range", 0.2)
    agent_cfg["entropy_loss_scale"] = training.get("entropy_coeff", 0.01)
    agent_cfg["value_loss_scale"] = training.get("value_loss_coeff", 0.5)
    agent_cfg["grad_norm_clip"] = training.get("max_grad_norm", 1.0)
    agent_cfg["rollouts"] = rollout_steps

    # 관측 전처리 (RunningStandardScaler)
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {
        "size": obs_dim,
        "device": device,
    }
    agent_cfg["value_preprocessor"] = RunningStandardScaler
    agent_cfg["value_preprocessor_kwargs"] = {
        "size": 1,
        "device": device,
    }

    # 체크포인트 저장 경로
    output_dir = training.get("output_dir", "training/rl/checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    agent_cfg["experiment"]["directory"] = output_dir
    agent_cfg["experiment"]["experiment_name"] = "so101_reach_ppo"
    agent_cfg["experiment"]["write_interval"] = training.get("log_freq", 10)
    agent_cfg["experiment"]["checkpoint_interval"] = training.get("save_freq", 100)

    # PPO 에이전트 생성
    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # 체크포인트 재로드
    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.exists():
            agent.load(str(resume_path))
            logger.info("체크포인트 로드 완료: %s", resume_path)
        else:
            logger.warning("체크포인트 파일을 찾을 수 없습니다: %s", resume_path)

    # --- 트레이너 설정 ---
    max_iterations = training.get("max_iterations", 1000)
    trainer_cfg = {
        "timesteps": max_iterations * rollout_steps,
        "headless": True,
    }

    trainer = SequentialTrainer(
        cfg=trainer_cfg,
        env=env,
        agents=agent,
    )

    logger.info("PPO 에이전트 구성 완료")
    logger.info("  학습률: %s", agent_cfg["learning_rate"])
    logger.info("  에포크: %d", agent_cfg["learning_epochs"])
    logger.info("  미니배치: %d", agent_cfg["mini_batches"])
    logger.info("  할인율: %s", agent_cfg["discount_factor"])
    logger.info("  GAE λ: %s", agent_cfg["lambda"])
    logger.info("  클리핑 범위: %s", agent_cfg["ratio_clip"])
    logger.info("  롤아웃 스텝: %d", rollout_steps)
    logger.info("  최대 반복: %d", max_iterations)
    logger.info("  체크포인트 경로: %s", output_dir)

    return agent, trainer


def run_training(agent, trainer):
    """강화학습을 실행한다.

    skrl 트레이너를 통해 PPO 에이전트를 학습한다.

    Args:
        agent: skrl PPO 에이전트
        trainer: skrl SequentialTrainer

    Raises:
        RuntimeError: 학습 중 오류 발생 시
    """
    logger.info("=== PPO 강화학습 시작 ===")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("학습이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error("학습 중 오류 발생: %s", e)
        raise

    logger.info("=== PPO 강화학습 완료 ===")


def main():
    """RL 학습 메인 엔트리포인트."""
    args = parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== skrl PPO RL Training Wrapper ===")

    # 1. 설정 로드
    config = load_config(args.config)

    # CLI 인자로 설정 오버라이드
    if args.max_iterations is not None:
        config["training"]["max_iterations"] = args.max_iterations
        logger.info("max_iterations 오버라이드: %d", args.max_iterations)

    if args.seed is not None:
        config["training"]["seed"] = args.seed
        logger.info("seed 오버라이드: %d", args.seed)

    # 2. 제어 파라미터 로드 및 교차검증
    control_params = load_control_params()
    validate_config(config, control_params)

    # 3. Dry-run 모드 처리
    if args.dry_run:
        env_cfg = config["environment"]
        training = config["training"]
        network = training.get("network", {})
        reward = config.get("reward", {})
        dr = config.get("domain_randomization", {})

        logger.info("=== Dry-run 모드: 설정 검증 완료 ===")
        logger.info("태스크: %s", env_cfg.get("task_name"))
        logger.info("병렬 환경 수: %d", args.num_envs or env_cfg.get("num_envs"))
        logger.info("알고리즘: %s", training.get("algorithm"))
        logger.info("학습률: %s", training.get("learning_rate"))
        logger.info("최대 반복: %d", training.get("max_iterations"))
        logger.info("롤아웃 스텝: %d", training.get("rollout_steps"))
        logger.info("네트워크: %s %s", network.get("type"), network.get("hidden_layers"))
        logger.info("활성화: %s", network.get("activation"))
        logger.info("보상 — reaching: %s, penalty: %s", reward.get("reaching_weight"), reward.get("action_penalty_weight"))
        logger.info("성공 임계값: %s m", reward.get("success_threshold"))
        logger.info("도메인 랜덤화: %s", "활성화" if dr.get("enabled") else "비활성화")
        logger.info("출력 경로: %s", training.get("output_dir"))
        logger.info("시드: %d", training.get("seed"))
        if args.resume:
            logger.info("재학습 체크포인트: %s", args.resume)
        logger.info("학습을 실행하려면 --dry-run 플래그를 제거하세요.")
        return

    # 4. Isaac Lab 환경 초기화
    env = setup_environment(
        config,
        num_envs=args.num_envs,
        headless=args.headless,
        task=args.task,
    )

    # 5. skrl PPO 에이전트 구성
    agent, trainer = setup_agent(
        config,
        env,
        resume_from=args.resume,
    )

    # 6. 학습 실행
    run_training(agent, trainer)

    # 7. 환경 정리
    env.close()
    logger.info("=== skrl PPO RL Training 완료 ===")


if __name__ == "__main__":
    main()
