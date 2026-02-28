#!/usr/bin/env python3
"""import_urdf_to_isaac.py — URDF→USD 변환 자동화 스크립트

목적:
    URDF 파일을 Isaac Sim의 URDF Importer를 통해 USD로 변환한다.
    robot_description/urdf/의 URDF를 입력받아 assets/usd/에 USD를 생성한다.

사용법:
    python scripts/import_urdf_to_isaac.py \\
        --input robot_description/urdf/so_arm101.urdf \\
        --output assets/usd/so_arm101.usd

필요 환경:
    - Isaac Sim Python 환경
    - URDF Importer extension 활성화

참고:
    - Isaac Sim URDF Importer 공식 문서: docs/references.md 참조
    - Isaac Lab 자산 Import 가이드: docs/references.md 참조

Phase: 4 (현재는 스텁)
상태: TODO — Phase 4에서 구현
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="URDF→USD 변환 (Isaac Sim URDF Importer)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 URDF 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/usd/",
        help="출력 USD 저장 경로 (파일 또는 디렉토리)",
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        default=True,
        help="베이스 링크 고정 여부",
    )
    parser.add_argument(
        "--merge-fixed-joints",
        action="store_true",
        default=False,
        help="고정 조인트 병합 여부",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== URDF → USD Converter ===")
    print(f"입력 URDF: {args.input}")
    print(f"출력 경로: {args.output}")
    print(f"베이스 고정: {args.fix_base}")
    print(f"고정 조인트 병합: {args.merge_fixed_joints}")
    print()
    print("[TODO] Phase 4에서 구현 예정")
    print()
    print("구현 예정 단계:")
    print("  1. Isaac Sim standalone 초기화")
    print("  2. URDF Importer config 설정")
    print("  3. URDF → USD 변환 실행")
    print("  4. 변환 결과 검증 (조인트 수, 메쉬 유무)")
    print("  5. USD 파일 저장")


if __name__ == "__main__":
    main()
