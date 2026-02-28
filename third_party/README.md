# third_party/ — 외부 레포지토리

이 폴더에는 외부 레포지토리를 clone 또는 git submodule로 추가한다.

## 서브모듈 추가 방법

```bash
# 예시: LeIsaac 추가
cd soarm_stack/
git submodule add https://github.com/LightwheelAI/leisaac.git third_party/leisaac

# 예시: 커뮤니티 Isaac Lab 태스크 추가
git submodule add https://github.com/MuammerBay/isaac_so_arm101.git third_party/isaac_so_arm101
```

## 규칙

- 외부 레포는 **특정 커밋 또는 태그**에 고정한다 (재현성).
- 서브모듈 추가 시 `docs/references.md`에도 URL/버전을 기록한다.
- 이 폴더의 내용은 `.gitignore`에 추가하지 않는다 (서브모듈은 git이 관리).
