# params/ — 물리/제어 파라미터

이 폴더는 시뮬레이션과 제어에 사용되는 모든 파라미터를 YAML 파일로 관리한다.

## 파일 목록

| 파일 | 내용 | Phase |
|------|------|-------|
| `physics.yaml` | 마찰, 댐핑, 질량, 접촉 파라미터 | Phase 3 (실제값) |
| `control.yaml` | 제어 주기, 게인, 지연 파라미터 | Phase 3 (실제값) |
| `data_pipeline.yaml` | 데이터 수집 파이프라인 파라미터 (fps, 카메라, 에피소드) | Phase 5 |

## 스키마 규칙

모든 파라미터는 다음 정보를 포함한다:

```yaml
parameter_name:
  value: 0.5          # 현재 값 (placeholder일 수 있음)
  unit: "N*m/rad"     # SI 단위
  range: [0.0, 10.0]  # 유효 범위 [min, max]
  description: "..."  # 파라미터 설명
```

## SSOT 원칙

- 스크립트에 매직넘버를 넣지 않는다. 항상 이 폴더의 YAML에서 읽는다.
- 파라미터 변경 시 YAML을 먼저 수정하고, CHANGELOG.md에 기록한다.
- Phase 0~2에서는 placeholder 값이다. Phase 3에서 실제값으로 채운다.
