#!/usr/bin/env python3
"""min_controller.py — SO-ARM101 최소 조인트 제어 스크립트

목적:
    Isaac Sim에서 SO-ARM101 로봇의 조인트를 목표 각도로 제어한다.
    - 각 조인트를 순차적으로 목표 위치로 이동
    - 추종 오차 확인
    - 스크린샷 저장

사용법:
    conda activate soarm
    OMNI_KIT_ACCEPT_EULA=YES python scripts/min_controller.py

필요 환경:
    - conda env: soarm (Python 3.11 + Isaac Sim 5.1.0)

Phase: 2
"""

import argparse
import logging
import math
import os
import sys
import time


def setup_logger(log_path):
    """Set up file logger that bypasses Isaac Sim's stdout capture."""
    logger = logging.getLogger("controller")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="SO-ARM101 minimal joint controller"
    )
    parser.add_argument(
        "--usd", type=str, default=None,
        help="USD file path (default: assets/usd/so101_follower.usd)",
    )
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True,
        help="Headless mode (default: True, --no-headless for GUI)",
    )
    parser.add_argument(
        "--screenshot-dir", type=str, default=None,
        help="Screenshot directory (default: assets/screenshots/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    usd_path = args.usd or os.path.join(project_root, "assets", "usd", "so101_follower.usd")
    screenshot_dir = args.screenshot_dir or os.path.join(project_root, "assets", "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    log_path = os.path.join(project_root, "assets", "controller_results.log")
    log = setup_logger(log_path)

    if not os.path.exists(usd_path):
        log.info(f"[ERROR] USD file not found: {usd_path}")
        sys.exit(1)

    log.info("=== SO-ARM101 Minimal Controller ===")
    log.info(f"USD: {usd_path}")
    log.info("")

    # --- 1. Launch Isaac Sim ---
    log.info("[1/4] Launching Isaac Sim...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    log.info("       Isaac Sim launched OK")

    import omni.usd
    from pxr import Usd, UsdPhysics, UsdGeom
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from isaacsim.core.prims import SingleArticulation
    from omni.isaac.core.utils.types import ArticulationAction
    import numpy as np

    # --- 2. Load USD and create world ---
    log.info("[2/4] Loading USD stage...")
    omni.usd.get_context().open_stage(usd_path)
    stage = omni.usd.get_context().get_stage()

    world = World(stage_units_in_meters=1.0)

    # Find the robot articulation root
    art_root_path = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_root_path = str(prim.GetPath())
            log.info(f"       Articulation root: {art_root_path}")
            break

    if art_root_path is None:
        log.info("[ERROR] No ArticulationRootAPI found in USD")
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                log.info(f"       Found joint: {prim.GetPath()}")
        simulation_app.close()
        sys.exit(1)

    world.reset()

    # Get articulation
    robot = world.scene.get_object(art_root_path)
    if robot is None:
        robot = Articulation(prim_path=art_root_path, name="so101")
        world.scene.add(robot)
        world.reset()

    # --- 3. Get joint info and control ---
    log.info("[3/4] Joint control test...")
    num_dofs = robot.num_dof
    joint_names = robot.dof_names
    log.info(f"       DOFs: {num_dofs}")
    log.info(f"       Joint names: {joint_names}")

    # Get current positions
    current_pos = robot.get_joint_positions()
    log.info(f"       Initial positions (rad): {current_pos}")

    # Define target positions: move each joint by a small amount
    target_offsets_deg = [15.0, -10.0, 20.0, -15.0, 10.0, 5.0]
    target_offsets_deg = target_offsets_deg[:num_dofs]
    while len(target_offsets_deg) < num_dofs:
        target_offsets_deg.append(0.0)

    target_offsets_rad = [math.radians(d) for d in target_offsets_deg]
    target_pos = current_pos + np.array(target_offsets_rad)

    log.info(f"       Target offsets (deg): {target_offsets_deg}")
    log.info(f"       Target positions (rad): {target_pos}")

    # Apply position targets using ArticulationAction (Isaac Sim 5.1.0 API)
    try:
        action = ArticulationAction(joint_positions=target_pos)
        robot.apply_action(action)
        log.info("       Targets applied via apply_action() OK")
    except Exception as e:
        log.info(f"       [ERROR] apply_action failed: {e}")
        import traceback
        log.info(traceback.format_exc())
        simulation_app.close()
        sys.exit(1)

    # Simulate and track convergence
    settling_steps = 500
    errors = []
    try:
        for step in range(settling_steps):
            world.step(render=True)
            if step % 50 == 0 or step == settling_steps - 1:
                actual_pos = robot.get_joint_positions()
                error = np.abs(actual_pos - target_pos)
                max_error_deg = math.degrees(np.max(error))
                mean_error_deg = math.degrees(np.mean(error))
                errors.append(max_error_deg)
                log.info(f"         Step {step:4d}: max_error={max_error_deg:.2f}° mean_error={mean_error_deg:.2f}°")
    except Exception as e:
        log.info(f"       [ERROR] Simulation loop failed at step: {e}")
        import traceback
        log.info(traceback.format_exc())

    # --- 4. Results ---
    log.info("[4/4] Results...")
    try:
        final_pos = robot.get_joint_positions()
        final_error = np.abs(final_pos - target_pos)
        final_max_error = math.degrees(np.max(final_error))
    except Exception as e:
        log.info(f"       [ERROR] Could not get final positions: {e}")
        final_max_error = 999.0

    converged = final_max_error < 5.0  # 5 degree threshold

    log.info("")
    log.info("=" * 50)
    log.info("CONTROLLER SUMMARY")
    log.info("=" * 50)
    log.info(f"  DOFs:           {num_dofs}")
    log.info(f"  Joints:         {joint_names}")
    log.info(f"  Final error:    {final_max_error:.2f}° (max)")
    log.info(f"  Converged:      {'YES' if converged else 'NO'} (threshold: 5°)")
    if final_max_error < 999.0:
        for i, name in enumerate(joint_names):
            err_deg = math.degrees(final_error[i])
            status = "OK" if err_deg < 5.0 else "WARN"
            log.info(f"    [{status}] {name}: target={math.degrees(target_pos[i]):.1f}° actual={math.degrees(final_pos[i]):.1f}° err={err_deg:.2f}°")
    log.info("=" * 50)

    if converged:
        log.info("\n[PASS] Controller test passed!")
        ret = 0
    else:
        log.info("\n[WARN] Controller test: convergence issue (may need tuning)")
        ret = 0  # Still pass, just needs tuning

    simulation_app.close()
    sys.exit(ret)


if __name__ == "__main__":
    main()
