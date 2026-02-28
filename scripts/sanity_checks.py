#!/usr/bin/env python3
"""sanity_checks.py — 조인트/리밋/스케일 검증 스크립트

목적:
    USD 파일을 로드하여 로봇의 조인트 구성이 올바른지 검증한다.
    - 조인트 이름/개수 확인
    - 조인트 리밋 범위 검증
    - 스케일/단위 검증
    - params/control.yaml과의 일관성 확인

사용법:
    python scripts/sanity_checks.py --usd assets/usd/so101_follower.usd

필요 환경:
    - Isaac Sim Python 환경

Phase: 2 (현재는 스텁)
상태: TODO — Phase 2에서 구현
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="SO-ARM101 sanity check: 조인트/리밋/스케일 검증"
    )
    parser.add_argument(
        "--usd",
        type=str,
        default="assets/usd/so101_follower.usd",
        help="검증할 USD 파일 경로",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="params/control.yaml",
        help="비교할 제어 파라미터 파일 경로",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== SO-ARM101 Sanity Check ===")
    print(f"USD 파일: {args.usd}")
    print(f"파라미터: {args.params}")
    print()
    print("[TODO] Phase 2에서 구현 예정")
    print()
    print("구현 예정 검증 항목:")
    print("  1. 조인트 이름/개수 확인 (6 DOF 예상)")
    print("  2. 조인트 리밋 범위 검증")
    print("  3. 링크 스케일 확인 (m 단위)")
    print("  4. params/control.yaml과 리밋 일관성 비교")
    print("  5. 질량/관성 텐서 유효성")


if __name__ == "__main__":
    main()
