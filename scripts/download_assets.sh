#!/usr/bin/env bash
# download_assets.sh — LeIsaac USD/씬 자산 + SO-ARM100 URDF 다운로드
#
# 목적:
#   LeIsaac v0.1.0 릴리즈에서 SO-ARM101 USD 파일과 씬 자산을 다운로드하고,
#   SO-ARM100 원본 레포에서 URDF/메쉬를 가져온다.
#   멱등성 보장 — 이미 존재하면 스킵한다.
#
# 사용법:
#   cd soarm_stack/
#   bash scripts/download_assets.sh
#
# Phase: 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# URLs from docs/references.md
USD_URL="https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"
SCENE_ZIP_URL="https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/kitchen_with_orange.zip"
SO_ARM100_REPO="https://github.com/TheRobotStudio/SO-ARM100.git"
LEISAAC_REPO="https://github.com/LightwheelAI/leisaac.git"

# Target paths
USD_DIR="$PROJECT_ROOT/assets/usd"
SCENE_DIR="$PROJECT_ROOT/assets/scenes/kitchen_with_orange"
URDF_DIR="$PROJECT_ROOT/robot_description/urdf"
MESH_DIR="$PROJECT_ROOT/robot_description/meshes"
THIRD_PARTY="$PROJECT_ROOT/third_party"

log_info()  { echo "[INFO]  $1"; }
log_ok()    { echo "[OK]    $1"; }
log_skip()  { echo "[SKIP]  $1 (already exists)"; }

echo "=== SO-ARM101 Asset Downloader ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Ensure directories exist
mkdir -p "$USD_DIR" "$SCENE_DIR" "$URDF_DIR" "$MESH_DIR" "$THIRD_PARTY"

# --- 1. Download USD robot file ---
if [ -f "$USD_DIR/so101_follower.usd" ]; then
    log_skip "so101_follower.usd"
else
    log_info "Downloading so101_follower.usd..."
    curl -L -o "$USD_DIR/so101_follower.usd" "$USD_URL"
    log_ok "so101_follower.usd ($(du -h "$USD_DIR/so101_follower.usd" | cut -f1))"
fi

# --- 2. Download and extract kitchen scene ---
if [ -d "$SCENE_DIR/kitchen_with_orange" ] && [ -f "$SCENE_DIR/kitchen_with_orange/scene.usd" ]; then
    log_skip "kitchen_with_orange scene"
else
    log_info "Downloading kitchen_with_orange.zip..."
    TMPZIP="$(mktemp /tmp/kitchen_with_orange.XXXXX.zip)"
    curl -L -o "$TMPZIP" "$SCENE_ZIP_URL"
    log_info "Extracting..."
    unzip -o "$TMPZIP" -d "$SCENE_DIR"
    rm -f "$TMPZIP"
    log_ok "kitchen_with_orange scene ($(du -sh "$SCENE_DIR" | cut -f1))"
fi

# --- 3. Clone SO-ARM100 for URDF/meshes ---
if [ -d "$THIRD_PARTY/SO-ARM100/.git" ]; then
    log_skip "SO-ARM100 repo"
else
    log_info "Cloning SO-ARM100..."
    git clone --depth 1 "$SO_ARM100_REPO" "$THIRD_PARTY/SO-ARM100"
    log_ok "SO-ARM100 cloned"
fi

# --- 4. Clone LeIsaac ---
if [ -d "$THIRD_PARTY/leisaac/.git" ]; then
    log_skip "LeIsaac repo"
else
    log_info "Cloning LeIsaac..."
    git clone --depth 1 "$LEISAAC_REPO" "$THIRD_PARTY/leisaac"
    log_ok "LeIsaac cloned"
fi

# --- 5. Copy URDF and meshes to robot_description (SSOT-A) ---
SO101_SIM="$THIRD_PARTY/SO-ARM100/Simulation/SO101"
if [ -f "$URDF_DIR/so101_new_calib.urdf" ]; then
    log_skip "URDF in robot_description"
else
    if [ -f "$SO101_SIM/so101_new_calib.urdf" ]; then
        cp "$SO101_SIM/so101_new_calib.urdf" "$URDF_DIR/"
        log_ok "URDF copied to robot_description/urdf/"
    else
        echo "[WARN]  URDF not found in SO-ARM100 repo"
    fi
fi

if ls "$MESH_DIR"/*.stl 1>/dev/null 2>&1; then
    log_skip "STL meshes in robot_description"
else
    if [ -d "$SO101_SIM/assets" ]; then
        cp "$SO101_SIM/assets"/*.stl "$MESH_DIR/" 2>/dev/null || true
        STL_COUNT=$(ls "$MESH_DIR"/*.stl 2>/dev/null | wc -l)
        log_ok "$STL_COUNT STL meshes copied to robot_description/meshes/"
    else
        echo "[WARN]  Mesh assets not found in SO-ARM100 repo"
    fi
fi

# --- Summary ---
echo ""
echo "=== Download Summary ==="
echo "  USD:    $(ls "$USD_DIR"/*.usd 2>/dev/null | wc -l) file(s) in assets/usd/"
echo "  Scene:  $(find "$SCENE_DIR" -name "*.usd" 2>/dev/null | wc -l) USD file(s) in assets/scenes/"
echo "  URDF:   $(ls "$URDF_DIR"/*.urdf 2>/dev/null | wc -l) file(s) in robot_description/urdf/"
echo "  Meshes: $(ls "$MESH_DIR"/*.stl 2>/dev/null | wc -l) STL file(s) in robot_description/meshes/"
echo "  Repos:  SO-ARM100, LeIsaac in third_party/"
echo ""
echo "Done!"
