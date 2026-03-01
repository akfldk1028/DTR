# params/ — 물리/제어 파라미터

이 폴더는 시뮬레이션과 제어에 사용되는 모든 파라미터를 YAML 파일로 관리한다.

## 파일 목록

| 파일 | 내용 | Phase |
|------|------|-------|
| `physics.yaml` | 마찰, 댐핑, 질량, 접촉 파라미터 | Phase 3 (실제값) |
| `control.yaml` | 제어 주기, 게인, 지연 파라미터 | Phase 3 (실제값) |

## 스키마 규칙

모든 파라미터는 다음 정보를 포함한다:

```yaml
parameter_name:
  value: 0.5          # 현재 값 (Phase 3 실측값 반영 완료)
  unit: "N*m/rad"     # SI 단위
  range: [0.0, 10.0]  # 유효 범위 [min, max]
  description: "..."  # 파라미터 설명
```

## SSOT 원칙

- 스크립트에 매직넘버를 넣지 않는다. 항상 이 폴더의 YAML에서 읽는다.
- 파라미터 변경 시 YAML을 먼저 수정하고, CHANGELOG.md에 기록한다.
- Phase 3에서 실제 측정값이 반영되었다. Phase 2의 `sanity_checks.py`와 `min_controller.py`로 검증된 값이다.
