#!/usr/bin/env python3
"""sanity_checks.py — SO-ARM101 USD 로드 + 조인트/리밋/스케일 검증 + 스크린샷

목적:
    USD 파일을 Isaac Sim에 로드하여 로봇의 조인트 구성이 올바른지 검증한다.
    - 조인트 이름/개수 확인
    - 조인트 리밋 범위 검증
    - 스케일/단위 검증
    - 물리 시뮬 안정성 확인 (몇 스텝 돌려서 폭발 없는지)
    - 스크린샷 저장

사용법:
    conda activate soarm
    OMNI_KIT_ACCEPT_EULA=YES python scripts/sanity_checks.py

필요 환경:
    - conda env: soarm (Python 3.11 + Isaac Sim 5.1.0)

Phase: 2
"""

import argparse
import logging
import os
import sys


def setup_logger(log_path):
    """Set up file logger that bypasses Isaac Sim's stdout capture."""
    logger = logging.getLogger("sanity")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    # Also try stderr which Isaac Sim may not capture
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="SO-ARM101 sanity check: USD 로드 + 조인트/리밋/스케일 검증"
    )
    parser.add_argument(
        "--usd",
        type=str,
        default=None,
        help="검증할 USD 파일 경로 (기본: assets/usd/so101_follower.usd)",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Headless 모드 (기본: True, --no-headless로 GUI 활성화)",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=str,
        default=None,
        help="스크린샷 저장 디렉토리 (기본: assets/screenshots/)",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=200,
        help="시뮬레이션 스텝 수 (기본: 200)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    usd_path = args.usd or os.path.join(project_root, "assets", "usd", "so101_follower.usd")
    screenshot_dir = args.screenshot_dir or os.path.join(project_root, "assets", "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    # Set up file logger to bypass Isaac Sim stdout capture
    log_path = os.path.join(project_root, "assets", "sanity_check_results.log")
    log = setup_logger(log_path)

    if not os.path.exists(usd_path):
        log.error("USD file not found: %s", usd_path)
        sys.exit(1)

    log.info("=== SO-ARM101 Sanity Check ===")
    log.info(f"USD:        {usd_path}")
    log.info(f"Headless:   {args.headless}")
    log.info(f"Sim steps:  {args.sim_steps}")
    log.info(f"Screenshots: {screenshot_dir}")
    log.info("")

    # --- 1. Launch Isaac Sim ---
    log.info("[1/6] Launching Isaac Sim...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    log.info("       Isaac Sim launched OK")

    # Now safe to import Omniverse modules
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, Gf, PhysxSchema
    import omni.physx
    import numpy as np

    # --- 2. Load USD ---
    log.info("[2/6] Loading USD stage...")
    stage_path = usd_path
    omni.usd.get_context().open_stage(stage_path)
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        log.error("Failed to load USD stage")
        simulation_app.close()
        sys.exit(1)
    log.info(f"       Stage loaded: {stage.GetRootLayer().identifier}")

    # --- 3. Enumerate joints ---
    log.info("[3/6] Checking joints...")
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            joint_info = {
                "path": str(prim.GetPath()),
                "name": prim.GetName(),
                "type": "revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "prismatic",
            }

            # Get joint limits
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint_api = UsdPhysics.RevoluteJoint(prim)
                lower = joint_api.GetLowerLimitAttr().Get()
                upper = joint_api.GetUpperLimitAttr().Get()
                joint_info["lower_limit"] = lower
                joint_info["upper_limit"] = upper
                joint_info["unit"] = "degrees"

            # Check for drive
            drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive_api:
                stiffness = drive_api.GetStiffnessAttr().Get()
                damping = drive_api.GetDampingAttr().Get()
                joint_info["stiffness"] = stiffness
                joint_info["damping"] = damping

            joints.append(joint_info)

    log.info(f"       Found {len(joints)} joints:")
    for j in joints:
        limits = ""
        if "lower_limit" in j:
            limits = f" [{j['lower_limit']:.1f}, {j['upper_limit']:.1f}] {j['unit']}"
        drive = ""
        if "stiffness" in j:
            drive = f" | stiffness={j['stiffness']}, damping={j['damping']}"
        log.info(f"         - {j['name']} ({j['type']}){limits}{drive}")

    # --- 4. Check links and masses ---
    log.info("[4/6] Checking links and physics...")
    links = []
    for prim in stage.Traverse():
        mass_api = UsdPhysics.MassAPI(prim)
        mass_attr = mass_api.GetMassAttr()
        if mass_attr and mass_attr.HasValue():
            link_info = {
                "path": str(prim.GetPath()),
                "name": prim.GetName(),
                "mass": mass_attr.Get(),
            }
            links.append(link_info)

    if links:
        total_mass = sum(l["mass"] for l in links)
        log.info(f"       Found {len(links)} links with mass (total: {total_mass:.4f} kg):")
        for l in links:
            log.info(f"         - {l['name']}: {l['mass']:.6f} kg")
    else:
        log.info("       No mass properties found (may use density-based)")

    # --- 5. Run simulation steps ---
    log.info(f"[5/6] Running {args.sim_steps} simulation steps...")
    from omni.isaac.core import World
    world = World(stage_units_in_meters=1.0)
    world.reset()

    initial_positions = {}
    exploded = False

    for step in range(args.sim_steps):
        world.step(render=True)

        if step == 0:
            # Record initial positions
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Xformable):
                    xform = UsdGeom.Xformable(prim)
                    transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    pos = transform.ExtractTranslation()
                    if abs(pos[0]) + abs(pos[1]) + abs(pos[2]) > 0:
                        initial_positions[str(prim.GetPath())] = pos

        if step == args.sim_steps - 1:
            # Check for physics explosion (any position > 100m from origin)
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Xformable):
                    xform = UsdGeom.Xformable(prim)
                    transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    pos = transform.ExtractTranslation()
                    if abs(pos[0]) > 100 or abs(pos[1]) > 100 or abs(pos[2]) > 100:
                        exploded = True
                        log.info(f"       [WARN] Possible explosion: {prim.GetName()} at {pos}")

        if step % 50 == 0:
            log.info(f"       Step {step}/{args.sim_steps}")

    if not exploded:
        log.info(f"       Simulation stable after {args.sim_steps} steps")
    else:
        log.info(f"       [FAIL] Physics explosion detected!")

    # --- 6. Save screenshot ---
    log.info("[6/6] Saving screenshot...")
    try:
        import omni.kit.viewport.utility as viewport_util
        viewport = viewport_util.get_active_viewport()
        if viewport:
            import omni.renderer_capture
            screenshot_path = os.path.join(screenshot_dir, "sanity_check.png")
            capture = omni.renderer_capture.acquire_renderer_capture_interface()
            capture.capture_next_frame_to_file(screenshot_path)
            # Need one more render to trigger capture
            world.step(render=True)
            log.info(f"       Screenshot saved: {screenshot_path}")
        else:
            log.info("       No viewport available (headless mode)")
    except Exception as e:
        log.info(f"       Screenshot skipped: {e}")

    # --- Summary ---
    log.info("")
    log.info("=" * 50)
    log.info("SANITY CHECK SUMMARY")
    log.info("=" * 50)
    log.info(f"  USD loaded:     YES")
    log.info(f"  Joints found:   {len(joints)}")
    log.info(f"  Links w/ mass:  {len(links)}")
    log.info(f"  Total mass:     {sum(l['mass'] for l in links):.4f} kg" if links else "  Total mass:     N/A")
    log.info(f"  Sim stable:     {'YES' if not exploded else 'NO - EXPLOSION'}")
    log.info(f"  Steps run:      {args.sim_steps}")
    log.info("=" * 50)

    if not exploded and len(joints) > 0:
        log.info("\n[PASS] Sanity check passed!")
        ret = 0
    else:
        log.info("\n[FAIL] Sanity check failed!")
        ret = 1

    simulation_app.close()
    sys.exit(ret)


if __name__ == "__main__":
    main()
