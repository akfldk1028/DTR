# References — 외부 링크 중앙 관리

> 모든 외부 URL은 이 파일에서 관리한다. 코드/문서에서는 이 파일을 참조한다.
> 링크가 바뀌면 이 파일만 수정하면 된다.

---

## 원본 / 기준점

| 이름 | URL | 버전/태그 | 용도 |
|------|-----|----------|------|
| SO-ARM100 원본 | https://github.com/TheRobotStudio/SO-ARM100 | main | 하드웨어/3D프린트 기준 |

## Isaac Sim / Lab

| 이름 | URL | 버전 | 용도 |
|------|-----|------|------|
| Seeed Isaac Sim 가이드 | https://wiki.seeedstudio.com/lerobot_so100m_isaacsim/ | - | Isaac Sim 연동 튜토리얼 |
| Seeed Isaac Lab 학습 | https://wiki.seeedstudio.com/training_soarm101_policy_with_isaacLab/ | - | Isaac Lab 학습 가이드 |
| Isaac Sim URDF Importer | https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html | 5.1.0 | URDF→USD 공식 문서 |
| Isaac Lab 자산 Import | https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html | main | 자산 import 공식 |

## LeRobot

| 이름 | URL | 버전 | 용도 |
|------|-----|------|------|
| LeRobot 메인 | https://github.com/huggingface/lerobot | main | 데이터/실물 파이프라인 |
| SO-101 문서 | https://huggingface.co/docs/lerobot/en/so101 | - | SO-101 HW 문서 |
| LeRobot v2 Dataset Format | https://huggingface.co/docs/lerobot/en/dataset_format | v2 | 데이터셋 포맷 공식 문서 |
| LeRobotDataset API | https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/lerobot_dataset.py | main | 데이터셋 API 레퍼런스 |

## LeIsaac

| 이름 | URL | 버전/태그 | 용도 |
|------|-----|----------|------|
| LeIsaac 메인 | https://github.com/LightwheelAI/leisaac | main | 시뮬 teleop→데이터→학습 |
| LeIsaac USD 릴리즈 | https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0 | v0.1.0 | 바로 쓰는 USD/씬 자산 |
| robot USD 직접 링크 | https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd | v0.1.0 | 로봇 USD 파일 |
| scene zip 직접 링크 | https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip | v0.1.0 | 씬 자산 zip |

## 커뮤니티

| 이름 | URL | 버전 | 용도 |
|------|-----|------|------|
| isaac_so_arm101 | https://github.com/MuammerBay/isaac_so_arm101 | main | 커뮤니티 Isaac Lab 태스크 (ArticulationCfg 참고) |
| GR00T N1.5 SO-101 | https://huggingface.co/nvidia/GR00T-N1.5-3B | - | NVIDIA GR00T 로봇 기반 모델, SO-101 post-training |
| XLeRobot | https://github.com/vector-wangel/xlerobot | main | 저비용 가정용 로봇 AI (SO-ARM 계열) |

## Context7 라이브러리 (공식 문서 조회용)

| 라이브러리 | Context7 ID | 스니펫 수 | 용도 |
|-----------|------------|----------|------|
| Isaac Sim | /isaac-sim/isaacsim/v5.1.0 | 304 | URDF Import, Articulation API |
| Isaac Lab | /isaac-sim/isaaclab | 2581 | ArticulationCfg, RL/IL training |
| Isaac Lab Docs | /websites/isaac-sim_github_io_isaaclab_main | 8382 | 전체 문서 (migration, env 설정) |
| LeRobot | /huggingface/lerobot | 1412 | 데이터셋 포맷, ACT/Diffusion/SmolVLA |
| LeIsaac | /lightwheelai/leisaac | 253 | 시뮬 teleop, HDF5→LeRobot 변환 |

## 핵심 논문 (arxiv)

| 이름 | arXiv ID | 연도 | Phase | 핵심 |
|------|----------|------|-------|------|
| Isaac Lab | 2511.04831 | 2025 | 3-5 | GPU 가속 시뮬레이션 프레임워크 공식 논문 |
| ACT/ALOHA | 2304.13705 | 2023 | 5-6 | 저비용 로봇 모방학습 베이스라인, LeRobot 지원 |
| Data Scaling Laws | 2410.18647 | 2024 | 4-6 | 데이터 다양성 > 양, 도메인 랜덤화 전략 |
| Real-is-Sim | 2504.03597 | 2025 | 5-7 | 디지털 트윈 기반 sim-to-real 파이프라인 |
| OpenVLA | 2406.09246 | 2024 | 6-7 | 7B 오픈소스 VLA, 소비자 GPU 파인튜닝 |
| VLA-RL | 2505.18719 | 2025 | 7 | VLA + 온라인 RL 개선 |
| RoboTwin | 2409.02920 | 2024 | 4 | AIGC 기반 합성 데이터 생성 |
