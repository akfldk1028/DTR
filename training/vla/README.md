# training/vla/ — Vision-Language-Action (VLA 확장)

## 개요

VLA 모델은 이미지 + 언어 지시(instruction)를 입력받아 로봇 액션을 출력하는 정책이다.
Isaac Lab은 VLA를 직접 학습하지 않고, **데이터 생성 + 평가 환경**으로 사용한다.

## I/O 계약 (Phase 7에서 구현)

```
입력:
  - image: RGB [H, W, 3] uint8       ← 카메라 관측
  - instruction: str                  ← 언어 지시 (예: "pick up the orange")

출력:
  - action: [6] float32              ← 목표 조인트 위치 [rad]
```

## 파이프라인

```
datasets/ (LeRobot + instruction 필드)
  → 외부 VLA 모델 학습 (PyTorch)
  → Isaac Sim/Lab에서 평가
    - instruction 입력 → 모델 추론 → action 실행 → 성공률 측정
```

## 역할 분담

| 컴포넌트 | 역할 |
|---------|------|
| Isaac Lab | 데이터 생성 + 평가 환경 ("세상") |
| VLA 모델 | 정책 ("뇌") — 별도 PyTorch 코드 |

## 현재 상태

Phase 0: 폴더 구조만 생성됨. Phase 7에서 구현 시작 (Phase 6 완료 후).
