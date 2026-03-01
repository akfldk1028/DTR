# datasets/ — 수집 데이터 (LeRobot v2 포맷)

이 폴더는 시뮬레이션에서 수집한 에피소드 데이터를 **LeRobot v2** 포맷으로 저장한다.
텔레오퍼레이션으로 기록한 조인트 위치, 카메라 이미지, 액션을 에피소드 단위로 관리한다.

> 외부 링크: `docs/references.md` 참조

---

## 디렉토리 구조

LeRobot v2 포맷은 다음과 같은 디렉토리 구조를 따른다:

```
datasets/
└── so101_teleop/                       # 데이터셋 이름 (repo_id 기반)
    ├── meta/                           # 메타데이터 (git 추적 대상)
    │   ├── info.json                   #   fps, features, shapes 등 데이터셋 정보
    │   ├── episodes.jsonl              #   에피소드별 메타데이터 (길이, 태스크 등)
    │   └── tasks.jsonl                 #   태스크 설명 목록
    ├── data/                           # 테이블 데이터 (git 제외)
    │   └── chunk-000/                  #   청크 디렉토리
    │       ├── episode_000000.parquet  #     에피소드 0 — 조인트 위치, 액션 등
    │       ├── episode_000001.parquet  #     에피소드 1
    │       └── ...
    └── videos/                         # 비디오 데이터 (git 제외)
        └── chunk-000/                  #   청크 디렉토리
            ├── observation.images.camera_episode_000000.mp4  # 에피소드 0 카메라 영상
            ├── observation.images.camera_episode_000001.mp4  # 에피소드 1 카메라 영상
            └── ...
```

### 파일 포맷

| 디렉토리 | 포맷 | 내용 | git 추적 |
|-----------|------|------|----------|
| `meta/` | JSON / JSONL | 데이터셋 메타데이터 (info, episodes, tasks) | ✅ 추적 |
| `data/chunk-NNN/` | Apache Parquet | 테이블 데이터 (조인트 위치, 액션, 타임스탬프) | ❌ 제외 |
| `videos/chunk-NNN/` | MP4 (H.264) | 카메라 이미지 시퀀스 (비디오 인코딩) | ❌ 제외 |

---

## Feature 스키마

SO-ARM101 데이터셋은 다음 3가지 feature로 구성된다:

| Feature 이름 | dtype | shape | 단위 | 설명 |
|-------------|-------|-------|------|------|
| `observation.state` | `float32` | `(6,)` | rad | 현재 조인트 위치 (6 DOF) |
| `observation.images.camera` | `video` | `(480, 640, 3)` | RGB | 카메라 이미지 (H×W×C) |
| `action` | `float32` | `(6,)` | rad | 목표 조인트 위치 (6 DOF) |

### 조인트 이름 (6 DOF)

`observation.state`와 `action`의 각 차원은 다음 조인트에 대응한다:

| 인덱스 | 조인트 이름 | 설명 |
|--------|------------|------|
| 0 | `shoulder_pan` | 어깨 회전 |
| 1 | `shoulder_lift` | 어깨 들기 |
| 2 | `elbow_flex` | 팔꿈치 굽힘 |
| 3 | `wrist_flex` | 손목 굽힘 |
| 4 | `wrist_roll` | 손목 회전 |
| 5 | `gripper` | 그리퍼 개폐 |

> 조인트 이름과 범위는 `params/data_pipeline.yaml`과 `params/control.yaml`에서 정의한다.

---

## 메타데이터 파일

### meta/info.json

데이터셋 전체 정보를 담는다:

```json
{
  "fps": 30,
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [6],
      "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper"]
    },
    "observation.images.camera": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "channels"]
    },
    "action": {
      "dtype": "float32",
      "shape": [6],
      "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper"]
    }
  }
}
```

### meta/episodes.jsonl

에피소드별 메타데이터를 한 줄씩 기록한다:

```jsonl
{"episode_index": 0, "length": 150, "task_index": 0}
{"episode_index": 1, "length": 200, "task_index": 0}
```

### meta/tasks.jsonl

태스크 설명을 기록한다 (Phase 7 VLA 학습에 사용):

```jsonl
{"task_index": 0, "task": "pick orange from table"}
```

---

## VLA 학습을 위한 task instruction 필드

VLA(Vision-Language-Action) 모델은 **자연어 지시문(instruction)**을 입력으로 받아
로봇 액션을 생성하는 정책이다. 데이터셋의 `task` 필드는 이 지시문을 에피소드와 연결하는
핵심 메타데이터이다.

> 상세 VLA 추론/평가 가이드: `training/vla/README.md` 참조

### task 필드 저장 흐름

1. **수집 시**: `dataset.save_episode(task="pick orange from table")` 호출
2. **tasks.jsonl 기록**: 고유한 태스크 문자열에 `task_index`가 부여되고 `meta/tasks.jsonl`에 저장
3. **episodes.jsonl 연결**: 각 에피소드는 `task_index`로 해당 태스크를 참조

```
save_episode(task="pick orange from table")
  → meta/tasks.jsonl:    {"task_index": 0, "task": "pick orange from table"}
  → meta/episodes.jsonl: {"episode_index": 0, "length": 150, "task_index": 0}
```

같은 태스크 문자열로 여러 에피소드를 저장하면 동일한 `task_index`를 공유한다.
다른 태스크 문자열을 사용하면 새로운 `task_index`가 생성된다:

```jsonl
# meta/tasks.jsonl — 태스크 목록
{"task_index": 0, "task": "pick orange from table"}
{"task_index": 1, "task": "place orange in bowl"}

# meta/episodes.jsonl — 에피소드별 태스크 연결
{"episode_index": 0, "length": 150, "task_index": 0}
{"episode_index": 1, "length": 200, "task_index": 0}
{"episode_index": 2, "length": 180, "task_index": 1}
```

### VLA 모델에서의 활용

VLA 모델은 학습 시 `task` 필드를 자연어 지시문으로 사용하여 조건부 정책을 학습한다:

```
학습 입력:  task (str) + observation.images.camera (480×640×3) + observation.state (6,)
학습 출력:  action (6,) — 6-DOF 관절 위치 타겟 [rad]
```

| 단계 | 역할 | task 필드 사용 |
|------|------|---------------|
| 데이터 수집 | `save_episode(task="...")` | 에피소드에 지시문 기록 |
| VLA 학습 | LeRobot DataLoader | `task` 텍스트를 언어 토큰으로 인코딩 |
| VLA 추론 | `model.predict(instruction, ...)` | 동일한 지시문을 입력으로 전달 |
| 시뮬 평가 | `eval_in_sim.py` | `--instruction` 인자로 지시문 지정 |

### task 설명 작성 가이드

- **자연어로 간결하게** 작성한다: `"pick orange from table"`, `"place cup on shelf"`
- **영어로** 작성한다 (VLA 사전학습 모델이 영어 기반)
- **행동 + 대상 + 위치** 형식을 권장한다: `"pick [object] from [location]"`
- 같은 동작을 수집하는 에피소드는 **동일한 task 문자열**을 사용한다

### 코드 예시

```python
# 데이터 수집 시 task 지정
dataset.save_episode(task="pick orange from table")

# CLI에서 task 지정
# python scripts/collect_data.py --task_description "pick orange from table"

# VLA 추론 시 동일한 task를 instruction으로 전달
from training.vla.inference import DummyVLA
model = DummyVLA()
action = model.predict(
    instruction="pick orange from table",  # ← task 필드와 동일
    image=image,
    state=state,
)
```

---

## 데이터 수집 (collect_data.py)

`scripts/collect_data.py`를 사용하여 Isaac Sim 텔레오퍼레이션에서 데이터를 수집한다.

### 기본 사용법

```bash
# 기본 파라미터로 1 에피소드 수집 (params/data_pipeline.yaml 기본값 사용)
python scripts/collect_data.py

# CLI 인자로 오버라이드
python scripts/collect_data.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --repo_id local/so101_teleop \
    --num_episodes 5 \
    --fps 30 \
    --num_envs 1

# 커스텀 태스크 설명 지정
python scripts/collect_data.py \
    --task_description "pick orange from table" \
    --num_episodes 10
```

### 수집 흐름

1. `params/data_pipeline.yaml`에서 기본 파라미터 로드
2. CLI 인자로 오버라이드 가능
3. `LeRobotDataset.create()`로 데이터셋 초기화
4. Isaac Sim 텔레오퍼레이션 환경 시작
5. 매 프레임마다 `dataset.add_frame()` 호출:
   - `observation.images.camera` — 카메라 RGB 이미지
   - `observation.state` — 현재 조인트 위치 [rad]
   - `action` — 목표 조인트 위치 [rad]
6. 에피소드 종료 시 `dataset.save_episode(task="...")` 호출
7. 수집 요약 로그 출력

### 필요 환경

- Isaac Sim 5.1.0 (`isaaclab` 패키지)
- LeRobot 0.4.4 (`lerobot` 패키지)
- conda env: `soarm`

---

## 데이터 검증 (validate_dataset.py)

`scripts/validate_dataset.py`를 사용하여 수집된 데이터셋의 무결성을 검증한다.

### 기본 사용법

```bash
# 데이터셋 구조 검증
python scripts/validate_dataset.py --repo_id local/so101_teleop

# 경로 직접 지정
python scripts/validate_dataset.py --dataset_path datasets/so101_teleop

# 시뮬레이션 리플레이 포함 검증 (Isaac Sim 필요)
python scripts/validate_dataset.py \
    --repo_id local/so101_teleop \
    --replay \
    --task_name LeIsaac-SO101-PickOrange-v0
```

### 검증 항목

| 검증 항목 | 설명 |
|-----------|------|
| 에피소드 수 | 최소 1개 이상의 에피소드 존재 확인 |
| 프레임 수 | 에피소드별 최소 1개 이상의 프레임 존재 확인 |
| Feature shape | `observation.state (6,)`, `action (6,)`, `observation.images.camera (480,640,3)` |
| Feature dtype | `float32`, `float32`, `video` |
| 메타 파일 | `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl` 존재 |
| Parquet 파일 | `data/chunk-*/` 디렉토리에 `.parquet` 파일 존재 |
| 비디오 파일 | `videos/chunk-*/` 디렉토리에 `.mp4` 파일 존재 |
| 리플레이 (선택) | 저장된 action 시퀀스로 시뮬 재실행, 상태 비교 |

### 검증 결과 예시

```
=== Validation Report ===
  [PASS] 에피소드 수 >= 1 — 총 3개 에피소드
  [PASS] 에피소드별 프레임 수 — episode 0: 150 frames; episode 1: 200 frames
  [PASS] Feature shape: observation.state — expected=(6,), actual=(6,)
  [PASS] Feature shape: action — expected=(6,), actual=(6,)
  [PASS] Feature dtype: observation.state — expected=float32, actual=float32
  [PASS] Parquet 파일 존재 (data/chunk-*/) — 3개 parquet 파일 발견
  [PASS] 비디오 파일 존재 (videos/chunk-*/) — 3개 비디오 파일 발견
---
Total: 7 | Passed: 7 | Failed: 0
```

---

## LeRobotDataset API 사용 예시

### 새 데이터셋 생성

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Feature 스키마 정의
features = {
    "observation.images.camera": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"],
    },
}

# 데이터셋 생성
dataset = LeRobotDataset.create(
    repo_id="local/so101_teleop",
    fps=30,
    features=features,
)
```

### 프레임 기록 및 에피소드 저장

```python
import numpy as np

# 에피소드 데이터 기록
for step in range(num_steps):
    image = camera.get_rgb()           # (480, 640, 3) uint8
    state = robot.get_joint_positions() # (6,) float32
    action = controller.get_action()    # (6,) float32

    dataset.add_frame({
        "observation.images.camera": image,
        "observation.state": state,
        "action": action,
    })

# 에피소드 저장 (task 설명 포함)
dataset.save_episode(task="pick orange from table")
```

### 저장된 데이터셋 로드

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 데이터셋 로드
dataset = LeRobotDataset(repo_id="local/so101_teleop")

# 기본 정보 확인
print(f"총 에피소드 수: {dataset.meta.total_episodes}")
print(f"FPS: {dataset.fps}")
print(f"Features: {list(dataset.meta.features.keys())}")

# 프레임 접근
frame = dataset[0]
state = frame["observation.state"]   # torch.Tensor, shape (6,)
action = frame["action"]             # torch.Tensor, shape (6,)
image = frame["observation.images.camera"]  # torch.Tensor
```

---

## 저장 규칙

- **대용량 데이터** (`data/`, `videos/`)는 `.gitignore`에 의해 git에서 제외한다.
- **메타데이터** (`meta/`)만 git에 포함하여 데이터셋 스키마의 재현성을 보장한다.
- **파라미터**는 `params/data_pipeline.yaml`에서 관리한다 (매직넘버 금지).
- LeRobot v2 포맷을 사용한다 (v1.6은 deprecated).
- `LeRobotDataset.create()` 시 features dict를 반드시 지정한다.
- `save_episode(task="...")` 호출 시 task 필드를 포함한다 (Phase 7 VLA 학습용).

---

## 관련 파일

| 파일 | 설명 |
|------|------|
| `scripts/collect_data.py` | Isaac Sim 텔레오퍼레이션 데이터 수집 |
| `scripts/validate_dataset.py` | 데이터셋 구조 검증 및 리플레이 |
| `params/data_pipeline.yaml` | 데이터 파이프라인 파라미터 (fps, shape, 경로 등) |
| `params/control.yaml` | 로봇 제어 파라미터 (조인트 범위 등) |
| `training/vla/README.md` | VLA 추론/평가 가이드 (inference I/O 계약, 모델 옵션) |
| `training/vla/inference.py` | VLA 추론 인터페이스 (VLAInference, DummyVLA 등) |
| `params/vla_eval.yaml` | VLA 평가 파라미터 |
| `docs/references.md` | 외부 링크 (LeRobot, LeIsaac 문서 등) |
