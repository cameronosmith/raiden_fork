"""Capture a teleop joint state where the wrist camera sees the exo ArUco board.

This is step 1 of wrist-camera calibration. The user teleoperates the arm into
a pose where the wrist camera can see one of the YAM base-mounted exoskeleton
ArUco boards. Both the scene camera and the wrist camera stream live to the
Rerun web viewer, so the user can visually confirm the board is in view from
the wrist. Pressing the YELLOW (top) leader button captures the current
follower joint state to a JSON file and exits.

The saved joint state is later replayed by the wrist-calibration script
(forthcoming) which commands the arm to this exact pose, holds for a few
seconds, captures a wrist + scene frame, and solves T_ee_to_cam in closed
form using the prior:

    T_ee_to_cam = inv(T_base_to_ee) @ T_base_to_board @ inv(T_cam_to_board)

where T_base_to_board comes from ``link_to_aruco_transform`` (a design constant
for the exo board) and T_cam_to_board comes from PnP on the wrist image.

Streams visualised:
  - ``scene/rgb_aruco`` — scene cam + ArUco overlay (confirms the rig is fine).
  - ``wrist/rgb_aruco`` — wrist cam + ArUco overlay (CRITICAL: this is what
    you watch while teleoping; press yellow only when the board is in view).
  - ``info`` — text panel with live "scene SEEN / wrist SEEN" status.
"""
import os
import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_RAIDEN_ROOT = Path(__file__).resolve().parents[2]
_EXO_DIR = _RAIDEN_ROOT / "third_party" / "exo_redo"
if str(_EXO_DIR) not in sys.path:
    sys.path.insert(0, str(_EXO_DIR))

from raiden.calibration.exo_calibrate import (
    _scene_serial_from_config,
    _open_zed_native,
    _grab_bgr,
    _init_rerun,
    _draw_aruco_plane,
    _to_rerun_jpeg,
)


_DEFAULT_OUT_DIR = Path.home() / ".config" / "raiden"


@dataclass
class _State:
    stop: bool = False
    scene_seen: bool = False
    wrist_seen: bool = False
    last_q: Optional[np.ndarray] = None


def _annotate(bgr, K, dist, exo_cfg):
    """Run aruco detection on bgr. Returns (annotated_bgr, seen_bool)."""
    from ExoConfigs.exoskeleton import link_to_aruco_transform  # noqa: F401
    from exo_utils import do_est_aruco_pose, ARUCO_DICT
    link_cfg = exo_cfg.links["larger_coarse_board"]
    aruco_board = exo_cfg.aruco_board_objects["larger_coarse_board"]
    board_length = link_cfg.board_length
    try:
        r = do_est_aruco_pose(
            bgr, ARUCO_DICT, aruco_board, board_length,
            cameraMatrix=K, distCoeffs=dist,
        )
    except Exception:
        return bgr, False
    if r == -1:
        return bgr, False
    annotated = r["pose_vis"]
    _draw_aruco_plane(annotated, r["est_aruco_pose"], board_length, K)
    return annotated, True


def _capture_loop(scene_cam, wrist_cam, K_s, dist_s, K_w, dist_w,
                  exo_cfg, rr, state, viz_width, jpeg_quality, viz_hz):
    log_interval = 1.0 / max(viz_hz, 1)
    last_log = 0.0
    frame_idx = 0
    while not state.stop:
        scene_bgr = _grab_bgr(scene_cam)
        wrist_bgr = _grab_bgr(wrist_cam)
        if scene_bgr is None or wrist_bgr is None:
            time.sleep(0.01)
            continue

        scene_ann, scene_seen = _annotate(scene_bgr, K_s, dist_s, exo_cfg)
        wrist_ann, wrist_seen = _annotate(wrist_bgr, K_w, dist_w, exo_cfg)
        state.scene_seen = scene_seen
        state.wrist_seen = wrist_seen

        now = time.monotonic()
        if now - last_log >= log_interval:
            last_log = now
            import rerun as _rr
            rr.set_time("step", sequence=frame_idx)
            rr.log("scene/rgb_aruco",
                   _to_rerun_jpeg(scene_ann, target_w=viz_width, quality=jpeg_quality))
            rr.log("wrist/rgb_aruco",
                   _to_rerun_jpeg(wrist_ann, target_w=viz_width, quality=jpeg_quality))
            scene_tag = "SEEN" if scene_seen else "not seen"
            wrist_tag = ("SEEN — ready to press YELLOW" if wrist_seen
                         else "NOT SEEN — teleop until visible")
            info = f"scene board: {scene_tag}    |    wrist board: {wrist_tag}"
            rr.log("info", _rr.TextDocument(info))
        frame_idx += 1


def run_capture(
    arm: str,
    scene_cam_name: str,
    wrist_cam_name: str,
    camera_config_file: str,
    output_file: str,
    vis_port: int,
    scene_resolution: str,
    wrist_resolution: str,
    viz_width: int,
    jpeg_quality: int,
    viz_hz: int,
):
    import rerun as rr
    from raiden.robot.controller import RobotController
    from ExoConfigs.yam_exo import YAM_BASE_ONLY_CONFIG

    out_path = Path(output_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scene_serial = _scene_serial_from_config(camera_config_file, scene_cam_name)
    wrist_serial = _scene_serial_from_config(camera_config_file, wrist_cam_name)
    print(f"Opening scene camera ({scene_cam_name}, serial={scene_serial}) at {scene_resolution}...")
    scene_cam, K_s, dist_s, _, _ = _open_zed_native(scene_serial, scene_resolution)
    print(f"Opening wrist camera ({wrist_cam_name}, serial={wrist_serial}) at {wrist_resolution}...")
    wrist_cam, K_w, dist_w, _, _ = _open_zed_native(wrist_serial, wrist_resolution)

    _init_rerun(vis_port)

    print(f"\nInitialising teleop ({arm} leader + {arm} follower)...")
    robot_controller = RobotController(
        use_right_leader=(arm == "right"),
        use_left_leader=(arm == "left"),
        use_right_follower=(arm == "right"),
        use_left_follower=(arm == "left"),
    )
    robot_controller.setup_for_teleop_recording()
    robot_controller.enable_estop()
    robot_controller.start_teleoperation()
    print(f"✓ Teleop active ({arm}).")
    print()
    print("=" * 70)
    print("  Move the LEADER until the wrist camera sees the exo ArUco board.")
    print("  Watch the `wrist/rgb_aruco` stream in your browser.")
    print("  Press the YELLOW (top) leader button to CAPTURE joint state + exit.")
    print("=" * 70)
    print()

    state = _State()
    th = threading.Thread(
        target=_capture_loop, daemon=True,
        args=(scene_cam, wrist_cam, K_s, dist_s, K_w, dist_w,
              YAM_BASE_ONLY_CONFIG, rr, state, viz_width, jpeg_quality, viz_hz),
    )
    th.start()

    follower = robot_controller.follower_r if arm == "right" else robot_controller.follower_l
    if follower is None:
        raise RuntimeError(f"No follower attached for arm {arm!r}; check RobotController setup.")

    captured = None
    try:
        while True:
            if robot_controller.check_button_press() is not None:
                q = follower.get_joint_pos()
                captured = {
                    "arm": arm,
                    "joint_state": [float(x) for x in q],
                    "n_joints": int(len(q)),
                    "wrist_board_seen_at_capture": bool(state.wrist_seen),
                    "scene_board_seen_at_capture": bool(state.scene_seen),
                    "scene_camera": scene_cam_name,
                    "wrist_camera": wrist_cam_name,
                    "scene_resolution": scene_resolution,
                    "wrist_resolution": wrist_resolution,
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                }
                break
            time.sleep(0.05)
    finally:
        state.stop = True
        time.sleep(0.2)
        try:
            scene_cam.close()
        except Exception:
            pass
        try:
            wrist_cam.close()
        except Exception:
            pass

    if captured is None:
        print("\nNo joint state captured (interrupted).")
    else:
        with open(out_path, "w") as f:
            json.dump(captured, f, indent=2)
        seen_tag = "✓" if captured["wrist_board_seen_at_capture"] else "⚠"
        n_joints = captured["n_joints"]
        print(f"\n{seen_tag} Saved {n_joints}-DoF joint state → {out_path}")
        if not captured["wrist_board_seen_at_capture"]:
            print("  WARNING: wrist camera did NOT see the exo board at capture time.")
            print("  Re-run and press yellow only while the wrist board is visible.")

    try:
        robot_controller.cleanup()
    except SystemExit:
        pass


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("right", "left"), default="right",
                   help="Which arm to teleop (right by default).")
    p.add_argument("--scene_cam", default="scene_camera",
                   help="Scene-camera name in camera.json.")
    p.add_argument("--wrist_cam", default=None,
                   help="Wrist-camera name in camera.json. Default: {arm}_wrist_camera.")
    p.add_argument("--camera_config_file",
                   default=str(Path.home() / ".config" / "raiden" / "camera.json"))
    p.add_argument("--output_file", default=None,
                   help="Output JSON path. Default: ~/.config/raiden/wrist_calibration_pose_{arm}.json")
    p.add_argument("--vis_port", type=int, default=9092)
    p.add_argument("--scene_resolution", default="HD1080",
                   choices=("HD720", "HD1080", "HD2K"))
    p.add_argument("--wrist_resolution", default="HD1080",
                   choices=("HD720", "HD1080", "HD2K"))
    p.add_argument("--viz_width", type=int, default=1280)
    p.add_argument("--jpeg_quality", type=int, default=75)
    p.add_argument("--viz_hz", type=int, default=10)
    args = p.parse_args()

    wrist_cam = args.wrist_cam or f"{args.arm}_wrist_camera"
    out = args.output_file or str(
        _DEFAULT_OUT_DIR / f"wrist_calibration_pose_{args.arm}.json"
    )
    run_capture(
        arm=args.arm,
        scene_cam_name=args.scene_cam,
        wrist_cam_name=wrist_cam,
        camera_config_file=args.camera_config_file,
        output_file=out,
        vis_port=args.vis_port,
        scene_resolution=args.scene_resolution,
        wrist_resolution=args.wrist_resolution,
        viz_width=args.viz_width,
        jpeg_quality=args.jpeg_quality,
        viz_hz=args.viz_hz,
    )


if __name__ == "__main__":
    main()
