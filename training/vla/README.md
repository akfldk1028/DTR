# training/vla/ — Vision-Language-Action (VLA 확장)

## 개요

VLA 모델은 **이미지 + 언어 지시(instruction) + 관절 상태(state)**를 입력받아
**6-DOF 로봇 액션**을 출력하는 정책이다.
Isaac Lab은 VLA를 직접 학습하지 않고, **데이터 생성 + 평가 환경**으로 사용한다.

이 모듈은 다음 기능을 제공한다:

- `VLAInference`: 모든 VLA 모델의 추상 인터페이스
- `DummyVLA`: 파이프라인 검증용 zero-action 베이스라인
- `SmolVLAWrapper`: LeRobot SmolVLA 모델 래퍼
- `eval_in_sim.py`: Isaac Sim closed-loop 평가 스크립트

---

## 추론 I/O 계약 (Inference Contract)

```
입력:
  - instruction: str                  ← 자연어 지시문 (예: "pick up the orange")
  - image: np.ndarray [480, 640, 3]   ← 카메라 RGB 이미지 (uint8)
  - state: np.ndarray [6]             ← 현재 관절 위치 (float32, rad)

출력:
  - action: np.ndarray [6]            ← 목표 관절 위치 (float32, rad)
```

### Feature 스키마

| Feature | dtype | shape | 단위 | 설명 |
|---------|-------|-------|------|------|
| `instruction` | `str` | — | — | 자연어 태스크 지시문 |
| `image` | `uint8` | `(480, 640, 3)` | RGB | 카메라 관측 이미지 (H×W×C) |
| `state` | `float32` | `(6,)` | rad | 현재 조인트 위치 (6 DOF) |
| `action` | `float32` | `(6,)` | rad | 목표 조인트 위치 (6 DOF) |

### 조인트 이름 (6 DOF)

`state`와 `action`의 각 차원은 다음 조인트에 대응한다:

| 인덱스 | 조인트 이름 | 설명 |
|--------|------------|------|
| 0 | `shoulder_pan` | 어깨 회전 |
| 1 | `shoulder_lift` | 어깨 들기 |
| 2 | `elbow_flex` | 팔꿈치 굽힘 |
| 3 | `wrist_flex` | 손목 굽힘 |
| 4 | `wrist_roll` | 손목 회전 |
| 5 | `gripper` | 그리퍼 개폐 |

---

## 클래스 계층 구조

```
VLAInference (ABC)              ← 추상 기본 클래스 (predict 계약 정의)
├── DummyVLA                    ← zero action 반환 (파이프라인 검증용)
└── SmolVLAWrapper              ← LeRobot SmolVLAPolicy 래퍼
                                   (lerobot >= 0.4.4 필요)

── 향후 확장 (params/vla_eval.yaml에 정의됨) ──
├── OpenVLAWrapper              ← OpenVLA 7B 래퍼 (미구현)
└── GR00TWrapper                ← NVIDIA GR00T N1.5 3B 래퍼 (미구현)
```

### 보조 클래스

```
EvalRunner                      ← closed-loop 시뮬 평가 실행기
└── EpisodeMetrics              ← 에피소드 단위 평가 결과

model_factory(model_type)       ← 문자열 이름으로 VLA 모델 인스턴스 생성
load_eval_params()              ← params/vla_eval.yaml 로드
```

---

## 모델 옵션

| 모델 | 클래스 | 파라미터 수 | 설명 | 상태 |
|------|--------|-----------|------|------|
| DummyVLA | `DummyVLA` | 0 | zero action 반환, 파이프라인 검증용 | ✅ 구현 완료 |
| SmolVLA | `SmolVLAWrapper` | 450M | LeRobot SmolVLA 트랜스포머 | ✅ 구현 완료 |
| OpenVLA | — | 7B | OpenVLA 7B (4bit 양자화 권장) | ⏳ 미구현 |
| GR00T | — | 3B | NVIDIA GR00T N1.5 | ⏳ 미구현 |

### 의존성 요구사항

| 모델 | 필수 패키지 | 설치 방법 |
|------|-----------|----------|
| DummyVLA | numpy | 기본 포함 |
| SmolVLA | lerobot >= 0.4.4 | `pip install lerobot` |
| OpenVLA | — | (향후 추가) |
| GR00T | — | (향후 추가) |

---

## 디렉토리 구조

```
training/vla/
├── __init__.py          # 패키지 초기화 (VLAInference, DummyVLA, SmolVLAWrapper 공개)
├── inference.py         # VLA 추론 인터페이스 + 모델 구현체
├── eval_in_sim.py       # Isaac Sim closed-loop 평가 스크립트
└── README.md            # 이 문서

params/
└── vla_eval.yaml        # VLA 평가 파라미터 (모델/평가/관측/성공기준 설정)
```

---

## 파이프라인

```
datasets/ (LeRobot + instruction 필드)
  → 외부 VLA 모델 학습 (PyTorch / LeRobot)
  → Isaac Sim/Lab에서 closed-loop 평가
    - 환경 리셋
    - observation(image + state) 획득
    - instruction + image + state → model.predict() → action
    - action → 시뮬레이션 적용
    - 성공률, 궤적 오차, 에피소드 길이 측정
```

---

## SmolVLA 학습

SmolVLA 학습은 LeRobot 프레임워크를 사용한다. SO-ARM101 데이터셋으로 파인튜닝하는 절차:

### 1. 데이터 준비

```bash
# 시뮬레이션 데이터 수집 (datasets/ 참조)
python scripts/collect_data.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --repo_id local/so101_teleop \
    --num_episodes 50 \
    --task_description "pick up the orange from the table"
```

### 2. SmolVLA 학습

```bash
# LeRobot SmolVLA 파인튜닝
python -m lerobot.scripts.train \
    --policy.type=smolvla \
    --dataset.repo_id=local/so101_teleop \
    --dataset.episodes='[0:50]' \
    --output_dir=outputs/smolvla_so101 \
    --steps=50000 \
    --batch_size=8 \
    --lr=1e-4 \
    --save_freq=10000
```

### 3. 체크포인트 확인

학습 완료 후 `outputs/smolvla_so101/` 디렉토리에 체크포인트가 저장된다.
이 경로를 `eval_in_sim.py`의 `--checkpoint` 인자로 전달한다.

---

## 평가 (eval_in_sim.py)

`training/vla/eval_in_sim.py`를 사용하여 Isaac Sim에서 VLA 모델의 closed-loop 평가를 수행한다.

### 기본 사용법

```bash
# 파이프라인 검증 (Isaac Sim 없이, mock 관측 사용)
python training/vla/eval_in_sim.py --model dummy --dry-run

# DummyVLA로 Isaac Sim 평가
python training/vla/eval_in_sim.py --model dummy --num-episodes 10

# SmolVLA 체크포인트로 평가
python training/vla/eval_in_sim.py \
    --model smolvla \
    --checkpoint outputs/smolvla_so101/checkpoint_50000 \
    --num-episodes 20

# 커스텀 지시문 + 파라미터 오버라이드
python training/vla/eval_in_sim.py \
    --model smolvla \
    --checkpoint outputs/smolvla_so101/checkpoint_50000 \
    --instruction "pick up the cube" \
    --num-episodes 50

# 파라미터 파일 기본값으로 실행
python training/vla/eval_in_sim.py
```

### CLI 인자

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--model` | str | vla_eval.yaml | VLA 모델 타입 (`dummy`, `smolvla`) |
| `--checkpoint` | str | vla_eval.yaml | 모델 체크포인트 경로 |
| `--instruction` | str | vla_eval.yaml | 태스크 지시문 |
| `--num-episodes` | int | vla_eval.yaml | 평가 에피소드 수 |
| `--dry-run` | flag | False | Isaac Sim 없이 mock 관측으로 파이프라인 검증 |
| `--vla-eval-yaml` | str | `params/vla_eval.yaml` | 파라미터 파일 경로 |

> CLI 인자는 `params/vla_eval.yaml` 기본값을 오버라이드한다.

### 평가 흐름

1. `params/vla_eval.yaml`에서 파라미터 로드
2. CLI 인자로 오버라이드
3. `model_factory()`로 VLA 모델 인스턴스 생성
4. `EvalRunner` 초기화 (Isaac Sim 환경 lazy init)
5. 에피소드 루프 실행:
   - 환경 리셋
   - 매 스텝: observation → `model.predict()` → action 적용 → sim step
   - 스텝별 궤적 오차 = `||action - state||₂` (L2 노름)
   - 에피소드 종료 시 집계
6. 전체 평가 결과 집계 및 로그 출력

### 평가 메트릭

| 메트릭 | 설명 | 계산 방법 |
|--------|------|----------|
| `success_rate` | 성공률 (0.0 ~ 1.0) | 성공 에피소드 수 / 전체 에피소드 수 |
| `mean_trajectory_error` | 평균 궤적 오차 (rad) | 에피소드별 평균 L2 노름의 전체 평균 |
| `mean_episode_length` | 평균 에피소드 길이 (steps) | 에피소드 길이의 평균 |

### 성공 기준

에피소드는 다음 두 조건을 모두 만족할 때 성공으로 판정한다:

1. `episode_length >= min_episode_length` (기본값: 10 스텝)
2. `trajectory_error <= trajectory_error_tolerance` (기본값: 0.05 rad)

### 출력 예시

```
=== VLA Evaluation Start ===
  Model:        DummyVLA
  Instruction:  pick up the orange from the table
  Episodes:     10
  Max steps:    500
  Dry-run:      True
Starting episode 0: instruction='pick up the orange from the table', max_steps=500
Episode 0 complete: length=500, trajectory_error=0.0000, success=True
...
=== Evaluation Summary ===
  Success rate:           100.00%
  Mean trajectory error:  0.0000
  Mean episode length:    500.0
  Episodes: 10 total, 10 success, 0 fail
```

---

## 코드 예제

### 1. DummyVLA 추론 (파이프라인 검증)

```python
import numpy as np
from training.vla.inference import DummyVLA

# 모델 생성
model = DummyVLA()

# mock 입력 생성
instruction = "pick up the orange"
image = np.zeros((480, 640, 3), dtype=np.uint8)
state = np.zeros(6, dtype=np.float32)

# 추론
action = model.predict(instruction, image, state)
# action.shape == (6,), dtype == float32, 모든 값 0.0
```

### 2. SmolVLA 추론

```python
import numpy as np
from training.vla.inference import SmolVLAWrapper

# 체크포인트에서 모델 로드
model = SmolVLAWrapper("outputs/smolvla_so101/checkpoint_50000")

# 실제 관측값으로 추론
instruction = "pick up the orange from the table"
image = camera.get_rgb()                    # shape (480, 640, 3), uint8
state = robot.get_joint_positions()         # shape (6,), float32

action = model.predict(instruction, image, state)
# action: 6-DOF 목표 관절 위치 [rad]
```

### 3. model_factory로 모델 생성

```python
from training.vla.eval_in_sim import model_factory

# 더미 모델
model = model_factory("dummy")

# SmolVLA 모델
model = model_factory("smolvla", checkpoint_path="path/to/checkpoint")
```

### 4. 프로그래밍 방식 평가

```python
from training.vla.eval_in_sim import EvalRunner, load_eval_params, model_factory

# 파라미터 로드 및 모델 생성
eval_params = load_eval_params()
model = model_factory("dummy")

# EvalRunner 생성 (dry-run 모드)
runner = EvalRunner(model=model, eval_params=eval_params, dry_run=True)

# 평가 실행
results = runner.run_evaluation(
    instruction="pick up the orange from the table",
    num_episodes=5,
)

print(f"성공률: {results['success_rate']:.2%}")
print(f"평균 궤적 오차: {results['mean_trajectory_error']:.4f}")
print(f"평균 에피소드 길이: {results['mean_episode_length']:.1f}")
```

### 5. 커스텀 VLA 모델 구현

```python
import numpy as np
from training.vla.inference import VLAInference

class MyCustomVLA(VLAInference):
    """커스텀 VLA 모델 래퍼 예시."""

    def __init__(self, model_path: str) -> None:
        # 모델 로드 로직
        self.model = load_my_model(model_path)

    def predict(
        self, instruction: str, image: np.ndarray, state: np.ndarray
    ) -> np.ndarray:
        """커스텀 모델로 행동을 예측한다.

        Args:
            instruction: 자연어 지시문.
            image: 카메라 이미지, shape (480, 640, 3), dtype uint8.
            state: 관절 상태, shape (6,), dtype float32.

        Returns:
            action: 6-DOF 관절 위치 타겟, shape (6,), dtype float32.
        """
        # predict 계약: (str, ndarray, ndarray) → ndarray (6,)
        action = self.model.infer(instruction, image, state)
        return np.array(action, dtype=np.float32)
```

---

## 파라미터 설정 (params/vla_eval.yaml)

평가에 사용되는 모든 파라미터는 `params/vla_eval.yaml`에 정의되어 있다.

### 주요 파라미터

| 섹션 | 파라미터 | 기본값 | 단위 | 설명 |
|------|---------|--------|------|------|
| `model` | `type` | `"dummy"` | — | VLA 모델 타입 |
| `model` | `checkpoint_path` | `""` | path | 체크포인트 경로 |
| `model` | `device` | `"cuda"` | — | 추론 디바이스 |
| `model` | `quantization` | `"none"` | — | 양자화 설정 |
| `evaluation` | `num_episodes` | `10` | count | 평가 에피소드 수 |
| `evaluation` | `max_steps_per_episode` | `500` | steps | 에피소드당 최대 스텝 |
| `evaluation` | `instruction` | `"pick up the orange..."` | — | 태스크 지시문 |
| `success_criteria` | `trajectory_error_tolerance` | `0.05` | rad | 궤적 오차 허용 범위 |
| `success_criteria` | `min_episode_length` | `10` | steps | 최소 에피소드 길이 |
| `isaac_sim` | `task` | `"LeIsaac-SO101-PickOrange-v0"` | — | 평가 환경 태스크 |
| `observation` | `image_height` | `480` | px | 이미지 세로 해상도 |
| `observation` | `image_width` | `640` | px | 이미지 가로 해상도 |
| `observation` | `state_dim` | `6` | dim | 관절 상태 차원 |
| `observation` | `action_dim` | `6` | dim | 액션 차원 |

---

## 역할 분담

| 컴포넌트 | 역할 |
|---------|------|
| Isaac Lab | 데이터 생성 + 평가 환경 ("세상") |
| VLA 모델 | 정책 ("뇌") — 별도 PyTorch / LeRobot 코드 |
| `inference.py` | VLA 모델 추론 인터페이스 + 구현체 |
| `eval_in_sim.py` | 시뮬레이션 closed-loop 평가 루프 |
| `params/vla_eval.yaml` | 평가 파라미터 중앙 관리 |

---

## 필요 환경

- Isaac Sim 5.1.0 (`isaaclab` 패키지) — 실제 평가 시
- NumPy
- LeRobot 0.4.4 (`lerobot` 패키지) — SmolVLA 사용 시
- conda env: `soarm`

> `--dry-run` 모드에서는 Isaac Sim과 LeRobot 없이 파이프라인 검증이 가능하다.

## 현재 상태

Phase 7: VLA 확장 구현 완료.
- ✅ `VLAInference` 추상 인터페이스
- ✅ `DummyVLA` zero-action 베이스라인
- ✅ `SmolVLAWrapper` LeRobot 통합
- ✅ `eval_in_sim.py` closed-loop 평가 스크립트
- ✅ `params/vla_eval.yaml` 평가 파라미터
- ⏳ OpenVLA / GR00T 래퍼 (향후 추가)
