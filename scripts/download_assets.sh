#!/usr/bin/env bash
# download_assets.sh — LeIsaac USD/씬 자산 다운로드
#
# 목적:
#   LeIsaac v0.1.0 릴리즈에서 SO-ARM101 USD 파일과 씬 자산을 다운로드한다.
#   다운로드된 파일은 assets/usd/ 및 assets/scenes/에 저장된다.
#
# 사용법:
#   cd soarm_stack/
#   bash scripts/download_assets.sh
#
# Phase: 1 (현재는 스텁)
# 상태: TODO — Phase 1에서 구현

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== SO-ARM101 Asset Downloader ==="
echo "Project root: $PROJECT_ROOT"
echo ""
echo "[TODO] Phase 1에서 구현 예정"
echo ""
echo "다운로드 예정 파일:"
echo "  1. assets/usd/so101_follower.usd"
echo "     출처: https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"
echo ""
echo "  2. assets/scenes/kitchen_with_orange/"
echo "     출처: https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip"
echo ""
echo "Phase 1 구현 시 이 스크립트가 실제 다운로드를 수행합니다."
