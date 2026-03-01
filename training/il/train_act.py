#!/usr/bin/env python3
"""train_act.py — ACT (Action Chunking with Transformers) 학습 래퍼 스크립트

목적:
    LeRobot ACT 정책의 학습을 프로젝트 컨벤션에 맞게 래핑한다.
    training/il/config.yaml에서 하이퍼파라미터를 로드하고,
    params/control.yaml에서 로봇 제어 파라미터를 교차검증한 뒤 학습을 실행한다.

    - config.yaml 기반 하이퍼파라미터 로드
    - params/control.yaml 교차검증 (조인트 리밋, 제어 주파수)
    - LeRobot ACT policy 학습 실행
    - 체크포인트 저장 (training/il/checkpoints/)

사용법:
    python training/il/train_act.py --config training/il/config.yaml
    python training/il/train_act.py --config training/il/config.yaml --output-dir training/il/checkpoints
    python training/il/train_act.py --config training/il/config.yaml --dry-run

필요 환경:
    - LeRobot 0.4.4 (lerobot 패키지)
    - PyTorch (GPU 권장)
    - conda env: soarm

참고:
    - ACT 논문: arxiv:2304.13705 (Learning Fine-Grained Bimanual Manipulation)
    - LeRobot: docs/references.md 참조

Phase: 6
상태: ACT IL baseline 학습
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
        description="ACT (Action Chunking with Transformers) 정책 학습"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/il/config.yaml",
        help="학습 설정 파일 경로 (default: training/il/config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="체크포인트 저장 경로 (미지정 시 config의 output_dir 사용)",
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
        help="랜덤 시드 (미지정 시 config의 seed 사용)",
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

    required_keys = ["policy", "training", "dataset", "robot"]
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
        dict: 제어 파라미터 딕셔너리
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
    policy = config["policy"]
    training = config["training"]
    robot = config["robot"]

    # --- 정책 아키텍처 검증 ---
    if policy.get("type") != "act":
        raise ValueError(f"지원하지 않는 정책 타입: {policy.get('type')} (act만 지원)")

    if policy.get("chunk_size", 0) <= 0:
        raise ValueError("chunk_size는 양의 정수여야 합니다")

    if policy.get("n_action_dims") != robot.get("dof"):
        raise ValueError(
            f"n_action_dims({policy.get('n_action_dims')})와 "
            f"robot.dof({robot.get('dof')})가 불일치합니다"
        )

    # --- 학습 설정 검증 ---
    if training.get("batch_size", 0) <= 0:
        raise ValueError("batch_size는 양의 정수여야 합니다")

    if training.get("total_steps", 0) <= 0:
        raise ValueError("total_steps는 양의 정수여야 합니다")

    if training.get("learning_rate", 0) <= 0:
        raise ValueError("learning_rate는 양수여야 합니다")

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

    # 제어 주파수 비교
    ctrl_freq = control_params.get("control_frequency", {}).get("value")
    if ctrl_freq is not None and robot.get("control_frequency") != ctrl_freq:
        logger.warning(
            "control_frequency 불일치 — config: %s, params/control.yaml: %s",
            robot.get("control_frequency"),
            ctrl_freq,
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


def setup_training(config, output_dir=None, resume_from=None, seed=None):
    """LeRobot ACT 학습 환경을 구성한다.

    Args:
        config: 학습 설정 딕셔너리
        output_dir: 체크포인트 저장 경로 (None이면 config에서 읽음)
        resume_from: 재학습 시 체크포인트 경로
        seed: 랜덤 시드 (None이면 config에서 읽음)

    Returns:
        dict: LeRobot 학습에 전달할 설정 딕셔너리
    """
    training = config["training"]
    policy = config["policy"]
    dataset = config["dataset"]

    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = training.get("output_dir", "training/il/checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("체크포인트 저장 경로: %s", output_dir)

    # 시드 설정
    if seed is None:
        seed = training.get("seed", 42)

    # 재학습 체크포인트
    if resume_from is None:
        resume_from = training.get("resume_from")

    # LeRobot 학습 설정 구성
    train_config = {
        # 정책
        "policy_type": policy["type"],
        "chunk_size": policy["chunk_size"],
        "n_obs_steps": policy["n_obs_steps"],
        "dim_model": policy["dim_model"],
        "n_heads": policy["n_heads"],
        "n_encoder_layers": policy["n_encoder_layers"],
        "n_decoder_layers": policy["n_decoder_layers"],
        "dim_feedforward": policy["dim_feedforward"],
        "dropout": policy["dropout"],
        "latent_dim": policy["latent_dim"],
        "kl_weight": policy["kl_weight"],
        # 학습
        "optimizer": training["optimizer"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "lr_scheduler": training["lr_scheduler"],
        "warmup_steps": training["warmup_steps"],
        "batch_size": training["batch_size"],
        "total_steps": training["total_steps"],
        "save_freq": training["save_freq"],
        "eval_freq": training["eval_freq"],
        "log_freq": training["log_freq"],
        "num_workers": training["num_workers"],
        "pin_memory": training["pin_memory"],
        "seed": seed,
        # 데이터셋
        "dataset_repo_id": dataset["repo_id"],
        "dataset_root": dataset["root"],
        "dataset_split": dataset["split"],
        # 출력
        "output_dir": output_dir,
        "resume_from": resume_from,
    }

    # 로깅 설정
    logging_cfg = config.get("logging", {})
    train_config["log_backend"] = logging_cfg.get("backend", "tensorboard")
    train_config["log_dir"] = logging_cfg.get("log_dir", "training/il/logs")
    os.makedirs(train_config["log_dir"], exist_ok=True)

    logger.info("학습 설정 구성 완료 (총 %d 스텝)", train_config["total_steps"])
    return train_config


def run_training(train_config):
    """LeRobot ACT 학습을 실행한다.

    LeRobot의 학습 API를 호출하여 ACT 정책을 학습한다.
    LeRobot이 설치되지 않은 경우 안내 메시지를 출력하고 종료한다.

    Args:
        train_config: setup_training()에서 구성한 학습 설정

    Raises:
        ImportError: LeRobot이 설치되지 않은 경우
    """
    try:
        import torch
    except ImportError:
        logger.error(
            "PyTorch가 설치되지 않았습니다. "
            "설치: pip install torch"
        )
        sys.exit(1)

    try:
        import lerobot  # noqa: F401
    except ImportError:
        logger.error(
            "LeRobot이 설치되지 않았습니다. "
            "설치 방법:\n"
            "  pip install lerobot\n"
            "  또는: git clone https://github.com/huggingface/lerobot && "
            "cd lerobot && pip install -e .\n"
            "자세한 내용: docs/references.md 참조"
        )
        sys.exit(1)

    # GPU 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("GPU를 사용할 수 없습니다. CPU로 학습합니다 (매우 느림).")
    else:
        logger.info("GPU 사용: %s", torch.cuda.get_device_name(0))

    # 재현성 설정
    seed = train_config["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("랜덤 시드: %d", seed)

    # LeRobot ACT 학습 실행
    logger.info("=== ACT 학습 시작 ===")
    logger.info("정책: %s", train_config["policy_type"])
    logger.info("데이터셋: %s", train_config["dataset_repo_id"])
    logger.info("배치 크기: %d", train_config["batch_size"])
    logger.info("총 스텝: %d", train_config["total_steps"])
    logger.info("학습률: %s", train_config["learning_rate"])
    logger.info("출력 경로: %s", train_config["output_dir"])

    _execute_lerobot_training(train_config, device)


def _execute_lerobot_training(train_config, device):
    """LeRobot 학습 API를 호출한다.

    LeRobot의 train() 함수 또는 CLI를 통해 ACT 학습을 실행한다.
    LeRobot API 구조에 따라 Hydra config 또는 직접 호출 방식을 사용한다.

    Args:
        train_config: 학습 설정 딕셔너리
        device: 학습 디바이스 ("cuda" 또는 "cpu")
    """
    try:
        from lerobot.scripts.train import train as lerobot_train

        # LeRobot train() 함수에 전달할 설정 구성
        # LeRobot은 Hydra 기반 설정을 사용하므로 CLI override 형태로 전달
        overrides = [
            f"policy.type={train_config['policy_type']}",
            f"dataset.repo_id={train_config['dataset_repo_id']}",
            f"training.batch_size={train_config['batch_size']}",
            f"training.steps={train_config['total_steps']}",
            f"training.save_freq={train_config['save_freq']}",
            f"training.lr={train_config['learning_rate']}",
            f"training.seed={train_config['seed']}",
            f"output_dir={train_config['output_dir']}",
            f"device={device}",
        ]

        if train_config.get("resume_from"):
            overrides.append(f"resume={train_config['resume_from']}")

        logger.info("LeRobot train() 호출 (overrides: %d개)", len(overrides))
        lerobot_train(overrides=overrides)

    except (ImportError, AttributeError) as e:
        logger.warning(
            "LeRobot train() 직접 호출 실패 (%s). CLI 방식으로 전환합니다.", e
        )
        _execute_lerobot_cli(train_config, device)

    except Exception as e:
        logger.error("LeRobot 학습 중 오류 발생: %s", e)
        raise


def _execute_lerobot_cli(train_config, device):
    """LeRobot CLI를 통해 학습을 실행한다.

    LeRobot의 Python API를 직접 사용할 수 없는 경우 subprocess로 CLI를 호출한다.

    Args:
        train_config: 학습 설정 딕셔너리
        device: 학습 디바이스
    """
    import subprocess

    cmd = [
        sys.executable, "-m", "lerobot.scripts.train",
        f"--policy.type={train_config['policy_type']}",
        f"--dataset.repo_id={train_config['dataset_repo_id']}",
        f"--training.batch_size={train_config['batch_size']}",
        f"--training.steps={train_config['total_steps']}",
        f"--training.save_freq={train_config['save_freq']}",
        f"--training.lr={train_config['learning_rate']}",
        f"--training.seed={train_config['seed']}",
        f"--output_dir={train_config['output_dir']}",
        f"--device={device}",
    ]

    if train_config.get("resume_from"):
        cmd.append(f"--resume={train_config['resume_from']}")

    logger.info("LeRobot CLI 실행: %s", " ".join(cmd[:3]) + " ...")

    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        logger.error("LeRobot CLI 학습 실패 (exit code: %d)", result.returncode)
        sys.exit(result.returncode)

    logger.info("LeRobot CLI 학습 완료")


def main():
    """ACT 학습 메인 엔트리포인트."""
    args = parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== ACT IL Training Wrapper ===")

    # 1. 설정 로드
    config = load_config(args.config)

    # 2. 제어 파라미터 로드 및 교차검증
    control_params = load_control_params()
    validate_config(config, control_params)

    # 3. 학습 환경 구성
    train_config = setup_training(
        config,
        output_dir=args.output_dir,
        resume_from=args.resume,
        seed=args.seed,
    )

    # 4. Dry-run 모드 처리
    if args.dry_run:
        logger.info("=== Dry-run 모드: 설정 검증 완료 ===")
        logger.info("정책 타입: %s", train_config["policy_type"])
        logger.info("데이터셋: %s", train_config["dataset_repo_id"])
        logger.info("배치 크기: %d", train_config["batch_size"])
        logger.info("총 스텝: %d", train_config["total_steps"])
        logger.info("학습률: %s", train_config["learning_rate"])
        logger.info("출력 경로: %s", train_config["output_dir"])
        logger.info("로그 경로: %s", train_config["log_dir"])
        logger.info("시드: %d", train_config["seed"])
        if train_config.get("resume_from"):
            logger.info("재학습 체크포인트: %s", train_config["resume_from"])
        logger.info("학습을 실행하려면 --dry-run 플래그를 제거하세요.")
        return

    # 5. 학습 실행
    run_training(train_config)
    logger.info("=== ACT IL Training 완료 ===")


if __name__ == "__main__":
    main()
