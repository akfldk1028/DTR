"""inference.py — VLA 모델 추론 인터페이스

목적:
    Vision-Language-Action(VLA) 모델의 추론 인터페이스를 정의한다.
    - VLAInference: 추상 기본 클래스 (predict 계약 정의)
    - DummyVLA: zero action을 반환하는 파이프라인 검증용 더미 모델
    - SmolVLAWrapper: LeRobot SmolVLAPolicy 래퍼

추론 계약 (Inference Contract):
    Input:
        instruction (str)       — 자연어 지시문 (예: "pick up the orange")
        image (np.ndarray)      — 카메라 관측 이미지 (480×640×3, uint8)
        state (np.ndarray)      — 로봇 관절 상태 (6,)
    Output:
        action (np.ndarray)     — 6-DOF 관절 위치 타겟 (6,)

사용법:
    from training.vla.inference import DummyVLA, SmolVLAWrapper

    # 파이프라인 검증 (더미)
    model = DummyVLA()
    action = model.predict("pick up orange", image, state)

    # SmolVLA 추론
    model = SmolVLAWrapper("path/to/checkpoint")
    action = model.predict("pick up orange", image, state)

Phase: 7
상태: 구현 완료 — VLA 추론 인터페이스
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

# LeRobot SmolVLA import is guarded so the module can be
# imported and used (e.g. DummyVLA) without this heavy dependency.
try:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    SMOLVLA_AVAILABLE = True
except ImportError:
    SmolVLAPolicy = None
    SMOLVLA_AVAILABLE = False

logger = logging.getLogger(__name__)


class VLAInference(ABC):
    """VLA 모델 추론 추상 기본 클래스.

    모든 VLA 모델 래퍼는 이 클래스를 상속하고
    predict() 메서드를 구현해야 한다.

    추론 계약:
        Input:  instruction (str) + image (480×640×3) + state (6,)
        Output: action (6,) — 6-DOF 관절 위치 타겟
    """

    @abstractmethod
    def predict(
        self, instruction: str, image: np.ndarray, state: np.ndarray
    ) -> np.ndarray:
        """자연어 지시문과 관측값으로부터 행동을 예측한다.

        Args:
            instruction: 자연어 지시문 (예: "pick up the orange").
            image: 카메라 관측 이미지, shape (480, 640, 3), dtype uint8.
            state: 로봇 관절 상태 벡터, shape (6,), dtype float32.

        Returns:
            action: 6-DOF 관절 위치 타겟, shape (6,), dtype float32.
        """
        raise NotImplementedError


class DummyVLA(VLAInference):
    """Zero-action baseline for pipeline verification.

    Instruction을 무시하고 zero action을 반환하는 더미 모델.
    End-to-end 파이프라인 검증 및 시뮬 평가 루프 테스트에 사용한다.
    """

    def predict(
        self, instruction: str, image: np.ndarray, state: np.ndarray
    ) -> np.ndarray:
        """모든 입력을 무시하고 zero action을 반환한다.

        Args:
            instruction: 자연어 지시문 (무시됨).
            image: 카메라 관측 이미지 (무시됨).
            state: 로봇 관절 상태 (무시됨).

        Returns:
            action: 영벡터, shape (6,), dtype float32.
        """
        logger.debug(
            "DummyVLA.predict() called — instruction=%r, returning zeros",
            instruction,
        )
        return np.zeros(6, dtype=np.float32)


class SmolVLAWrapper(VLAInference):
    """LeRobot SmolVLAPolicy 래퍼.

    LeRobot의 SmolVLAPolicy를 VLAInference 인터페이스로 감싼다.
    체크포인트에서 모델을 로드하고, predict() 호출 시
    SmolVLAPolicy.select_action()을 위임한다.

    필요 환경:
        - lerobot >= 0.4.4
        - SmolVLA 체크포인트 (학습 완료)
    """

    def __init__(self, checkpoint_path: str) -> None:
        """SmolVLA 체크포인트에서 모델을 로드한다.

        Args:
            checkpoint_path: SmolVLA 체크포인트 디렉토리 경로.

        Raises:
            RuntimeError: LeRobot SmolVLA가 설치되어 있지 않은 경우.
        """
        if not SMOLVLA_AVAILABLE:
            raise RuntimeError(
                "LeRobot SmolVLA is not available. "
                "Install with: pip install lerobot"
            )

        logger.info(
            "Loading SmolVLA checkpoint: %s", checkpoint_path
        )
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
        logger.info("SmolVLA model loaded successfully")

    def predict(
        self, instruction: str, image: np.ndarray, state: np.ndarray
    ) -> np.ndarray:
        """SmolVLA 모델로 행동을 예측한다.

        Args:
            instruction: 자연어 지시문 (예: "pick up the orange").
            image: 카메라 관측 이미지, shape (480, 640, 3), dtype uint8.
            state: 로봇 관절 상태 벡터, shape (6,), dtype float32.

        Returns:
            action: 6-DOF 관절 위치 타겟, shape (6,), dtype float32.
        """
        observation = {
            "observation.images.camera": image,
            "observation.state": state,
            "task": instruction,
        }
        action = self.policy.select_action(observation)
        logger.debug(
            "SmolVLAWrapper.predict() — instruction=%r, action=%s",
            instruction,
            action,
        )
        return action
