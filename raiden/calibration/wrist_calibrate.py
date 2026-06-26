"""Wrist-camera calibration via live teleop + 5-press freeze cycle.

Interactive flow:
  yellow#1 FREEZE 1: stop teleop; clear history; accumulate aruco detections
                     for the calibration solve. Save wrist+scene snapshot
                     and joint state to ``<out>/freeze1/``.
  yellow#2 UNFREEZE: resume teleop. (Calibration data is locked in from
                     freeze 1; no more accumulation past this point.)
  yellow#3 FREEZE 2: stop teleop. Save a SECOND snapshot (wrist+scene+state)
                     to ``<out>/freeze2/``. No new aruco accumulation —
                     this snapshot is for cross-validation only.
  yellow#4 UNFREEZE: resume teleop. User can teleop to a safe end pose.
  yellow#5 END:      solve T_ee_cam from freeze 1's accumulated history,
                     save calibration, route current → INTERMEDIATE → HOME.

Math:
    T_ee_cam = inv(T_base_ee_freeze1) @ T_base_aruco @ inv(T_cam_aruco_freeze1)

The point of the two-pose flow: snapshot 1 supplies the calibration data
(FK + aruco PnP at the SAME static pose). Snapshot 2 is for offline
cross-validation: using the calibrated T_ee_cam, FK at pose 2 should
predict aruco positions in the second wrist image with low reprojection
error. That tests whether T_ee_cam is a true rigid mount (pose-independent).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    _PoseHistory,
    _solve_multiframe_pose,
)


_DEFAULT_OUT_DIR = Path.home() / ".config" / "raiden"
_VALIDATION_OUT_BASE = Path("/home/robot-lab/cameron/wrist_validation")
DOF = 7  # 6 arm joints + 1 gripper

INTERMEDIATE_POSE_RIGHT = np.array(
    [0.0097, 1.9213, 1.4582, -0.4324, 0.5880, -0.1459, 0.9923],
    dtype=np.float64,
)
INTERMEDIATE_POSE_LEFT = INTERMEDIATE_POSE_RIGHT.copy()
HOME_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class _LoopState:
    stop: bool = False
    frozen: bool = False         # arm is held (informational)
    accumulate: bool = False     # actively pooling aruco detections to history
    last_detection: Optional[dict] = None
    history: Optional[_PoseHistory] = None
    status_text: str = "TELEOP"


def _fk_base_to_ee(joint_state_7: np.ndarray) -> np.ndarray:
    from i2rt.robots.kinematics import Kinematics
    from raiden._xml_paths import get_yam_4310_linear_xml_path
    global _KIN_CACHE
    try:
        kin = _KIN_CACHE  # type: ignore[name-defined]
    except NameError:
        kin = Kinematics(get_yam_4310_linear_xml_path(), site_name="grasp_site")
        globals()["_KIN_CACHE"] = kin
    q = np.zeros(8, dtype=np.float64)
    q[:6] = joint_state_7[:6]
    return np.asarray(kin.fk(q), dtype=np.float64)


def _smooth_move(robot, target_7: np.ndarray, max_vel_rad_s: float,
                 min_move_s: float = 0.5, steps: int = 100) -> float:
    from raiden.robot.controller import smooth_move_joints
    current = np.asarray(robot.get_joint_pos(), dtype=np.float64)
    target = np.asarray(target_7, dtype=np.float64)
    delta = float(np.max(np.abs(target - current)))
    t_s = max(min_move_s, delta / max(max_vel_rad_s, 1e-6))
    print(f"    smooth_move dmax={delta:.3f} rad -> {t_s:.2f}s")
    smooth_move_joints(robot, target, time_interval_s=t_s, steps=steps)
    return t_s


def _capture_loop(scene_cam, wrist_cam, K_s, dist_s, K_w, dist_w,
                  exo_cfg, rr, state: _LoopState, viz_width: int,
                  jpeg_quality: int, viz_hz: int) -> None:
    """Daemon: detect aruco on wrist each frame; accumulate to history ONLY
    when frozen (so all pooled correspondences share the same FK). Log both
    annotated frames + status text to rerun.
    """
    from ExoConfigs.exoskeleton import link_to_aruco_transform  # noqa: F401
    from exo_utils import do_est_aruco_pose, ARUCO_DICT
    link_cfg = exo_cfg.links["larger_coarse_board"]
    aruco_board = exo_cfg.aruco_board_objects["larger_coarse_board"]
    board_length = link_cfg.board_length

    log_interval = 1.0 / max(viz_hz, 1)
    last_log = 0.0
    frame_idx = 0
    while not state.stop:
        scene_bgr = _grab_bgr(scene_cam)
        wrist_bgr = _grab_bgr(wrist_cam)
        if scene_bgr is None or wrist_bgr is None:
            time.sleep(0.01)
            continue

        scene_ann = scene_bgr.copy()
        try:
            rs = do_est_aruco_pose(scene_bgr, ARUCO_DICT, aruco_board, board_length,
                                   cameraMatrix=K_s, distCoeffs=dist_s)
            if rs != -1:
                scene_ann = rs["pose_vis"]
                _draw_aruco_plane(scene_ann, rs["est_aruco_pose"], board_length, K_s)
        except Exception:
            pass

        wrist_ann = wrist_bgr.copy()
        wrist_seen = False
        try:
            rw = do_est_aruco_pose(wrist_bgr, ARUCO_DICT, aruco_board, board_length,
                                   cameraMatrix=K_w, distCoeffs=dist_w)
            if rw != -1:
                wrist_ann = rw["pose_vis"]
                _draw_aruco_plane(wrist_ann, rw["est_aruco_pose"], board_length, K_w)
                wrist_seen = True
                obj_cam, img_pts = rw["obj_img_pts"]
                rvec, tvec_pre = rw["rtvec"]
                R_mat = cv2.Rodrigues(rvec)[0]
                center_offset_board = np.array(
                    [board_length / 2, board_length / 2, 0], dtype=np.float64
                )
                tvec_corner = tvec_pre - R_mat.dot(center_offset_board)
                obj_board = (R_mat.T @ (obj_cam.T - tvec_corner.reshape(3, 1))).T
                proj, _ = cv2.projectPoints(
                    obj_cam.astype(np.float32), rvec, tvec_pre, K_w, dist_w
                )
                resid_px = float(np.linalg.norm(
                    proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1
                ).mean())
                T_cam_aruco = rw["est_aruco_pose"]
                # Only accumulate during freeze #1 — keeps the multi-frame
                # pool consistent with a single FK at solve time. Freeze #2's
                # detections are NOT pooled (it's a separate cross-validation
                # snapshot, not part of the calibration data).
                if state.accumulate and state.history is not None:
                    state.history.add(
                        obj_pts=obj_board.astype(np.float32),
                        img_pts=img_pts.astype(np.float32),
                        residual_px=resid_px,
                        single_frame_t=T_cam_aruco[:3, 3],
                    )
                state.last_detection = {
                    "T_cam_aruco": T_cam_aruco,
                    "resid_px": resid_px,
                    "n_markers": int(len(obj_cam) // 4),
                }
        except Exception:
            pass

        now = time.monotonic()
        if now - last_log >= log_interval:
            last_log = now
            import rerun as _rr
            rr.set_time("step", sequence=frame_idx)
            rr.log("scene/rgb_aruco",
                   _to_rerun_jpeg(scene_ann, target_w=viz_width, quality=jpeg_quality))
            rr.log("wrist/rgb_aruco",
                   _to_rerun_jpeg(wrist_ann, target_w=viz_width, quality=jpeg_quality))
            seen = "SEEN" if wrist_seen else "not seen"
            hist = state.history.stats() if state.history is not None else ""
            rr.log("info", _rr.TextDocument(
                f"[{state.status_text}]  wrist board: {seen}    |    {hist}"
            ))
        frame_idx += 1


def _save_calibration_result(
    output_file: Path, wrist_cam_name: str, arm: str,
    K: np.ndarray, dist: np.ndarray, T_ee_cam: np.ndarray,
    resolution: str, n_used: int, mean_resid_px: float,
    joint_state: List[float],
) -> None:
    cal: dict = {}
    if output_file.exists():
        with open(output_file) as f:
            cal = json.load(f)
    cal.setdefault("version", "1.0")
    cal["timestamp"] = datetime.now().isoformat(timespec="seconds")
    cal.setdefault("cameras", {})
    cal["cameras"][wrist_cam_name] = {
        "type": "wrist_one_shot",
        "method": "exo_board_pnp_at_known_joint_state",
        "arm": arm,
        "resolution": resolution,
        "intrinsics": {
            "camera_matrix": K.tolist(),
            "distortion_coeffs": [float(v) for v in np.asarray(dist).reshape(-1)[:5]],
        },
        "num_frames_used": int(n_used),
        "mean_reproj_residual_px": float(mean_resid_px),
        "capture_joint_state": [float(x) for x in joint_state],
        "extrinsics": {
            "success": True,
            "rotation_matrix": T_ee_cam[:3, :3].tolist(),
            "translation_vector": T_ee_cam[:3, 3].tolist(),
            "reference_frame": f"{arm}_grasp_site",
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\n  Saved wrist calibration -> {output_file}")
    print(f"      cam_in_ee xyz = {T_ee_cam[:3, 3]}")
    print(f"      reference_frame = {arm}_grasp_site")
    print(f"      n_frames_used = {n_used},  mean_resid = {mean_resid_px:.2f} px")


def _save_freeze_snapshot(
    out_dir: Path, scene_bgr: np.ndarray, wrist_bgr: np.ndarray,
    achieved_q: List[float], arm: str,
    scene_cam_name: str, scene_serial: int, K_s: np.ndarray, dist_s: np.ndarray, scene_res: str,
    wrist_cam_name: str, wrist_serial: int, K_w: np.ndarray, dist_w: np.ndarray, wrist_res: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "scene.png"), scene_bgr)
    cv2.imwrite(str(out_dir / "wrist.png"), wrist_bgr)
    state_doc = {
        "arm": arm,
        "joint_state_achieved": achieved_q,
        "scene_camera": {
            "name": scene_cam_name, "serial": scene_serial,
            "K": K_s.tolist(), "dist": dist_s.tolist(),
            "resolution": scene_res,
        },
        "wrist_camera": {
            "name": wrist_cam_name, "serial": wrist_serial,
            "K": K_w.tolist(), "dist": dist_w.tolist(),
            "resolution": wrist_res,
        },
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }
    with open(out_dir / "state.json", "w") as f:
        json.dump(state_doc, f, indent=2)
    print(f"  Saved freeze snapshot -> {out_dir}")


def run_wrist_calibrate(
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
    max_joint_vel_rad_s: float,
    history_size: int,
    top_k: int,
):
    import rerun as rr
    from raiden.robot.controller import RobotController
    from ExoConfigs.yam_exo import YAM_BASE_ONLY_CONFIG
    from ExoConfigs.exoskeleton import link_to_aruco_transform

    # --- Cameras
    scene_serial = _scene_serial_from_config(camera_config_file, scene_cam_name)
    wrist_serial = _scene_serial_from_config(camera_config_file, wrist_cam_name)
    print(f"Opening scene camera ({scene_cam_name}, serial={scene_serial}) at {scene_resolution}...")
    scene_cam, K_s, dist_s, W_s, H_s = _open_zed_native(scene_serial, scene_resolution)
    print(f"Opening wrist camera ({wrist_cam_name}, serial={wrist_serial}) at {wrist_resolution}...")
    wrist_cam, K_w, dist_w, W_w, H_w = _open_zed_native(wrist_serial, wrist_resolution)

    _init_rerun(vis_port)

    # --- Robot
    print(f"\nInitialising robot ({arm} leader + {arm} follower)...")
    rc = RobotController(
        use_right_leader=(arm == "right"),
        use_left_leader=(arm == "left"),
        use_right_follower=(arm == "right"),
        use_left_follower=(arm == "left"),
    )
    rc.setup_for_teleop_recording()
    rc.enable_estop()
    rc.start_teleoperation()

    follower = rc.follower_r if arm == "right" else rc.follower_l
    intermediate_pose = (
        INTERMEDIATE_POSE_RIGHT if arm == "right" else INTERMEDIATE_POSE_LEFT
    )

    history = _PoseHistory(max_size=history_size)
    state = _LoopState(history=history, status_text="TELEOP — drive leader until wrist sees board")

    th = threading.Thread(
        target=_capture_loop, daemon=True,
        args=(scene_cam, wrist_cam, K_s, dist_s, K_w, dist_w,
              YAM_BASE_ONLY_CONFIG, rr, state, viz_width, jpeg_quality, viz_hz),
    )
    th.start()

    print()
    print("=" * 72)
    print("  Teleop the leader until the WRIST camera clearly sees the aruco board.")
    print("  Yellow #1 -> FREEZE 1 (calibrate). Accumulate aruco -> calibration solve.")
    print("  Yellow #2 -> UNFREEZE. Drive to a SECOND aruco-visible pose.")
    print("  Yellow #3 -> FREEZE 2 (verify). Snapshot only — not pooled into solve.")
    print("  Yellow #4 -> UNFREEZE. Drive somewhere safe to end.")
    print("  Yellow #5 -> END (solve from freeze 1, save, return home).")
    print("=" * 72)
    print()

    press_count = 0
    freeze1_q: Optional[np.ndarray] = None
    freeze1_dir: Optional[Path] = None
    freeze2_q: Optional[np.ndarray] = None
    freeze2_dir: Optional[Path] = None
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = _VALIDATION_OUT_BASE / f"wrist_calib_{session_ts}"

    def _capture_freeze(label: str):
        """Grab synchronized scene+wrist frames + joint state into <session>/<label>/."""
        achieved_q = list(map(float, follower.get_joint_pos()))
        out_dir = session_dir / label
        scene_now = _grab_bgr(scene_cam)
        wrist_now = _grab_bgr(wrist_cam)
        if scene_now is not None and wrist_now is not None:
            _save_freeze_snapshot(
                out_dir, scene_now, wrist_now, achieved_q, arm,
                scene_cam_name, scene_serial, K_s, dist_s, scene_resolution,
                wrist_cam_name, wrist_serial, K_w, dist_w, wrist_resolution,
            )
        return np.array(achieved_q), out_dir

    try:
        while True:
            if rc.check_button_press() is not None:
                press_count += 1
                if press_count == 1:
                    # FREEZE 1 — start accumulating for the calibration solve.
                    print("\n[yellow #1] FREEZE 1 (calibrate) — stopping teleop")
                    rc.stop_teleoperation()
                    time.sleep(0.1)
                    history.entries = []  # fresh accumulation
                    state.frozen = True
                    state.accumulate = True
                    state.status_text = "FROZEN 1 — accumulating aruco detections"
                    freeze1_q, freeze1_dir = _capture_freeze("freeze1")
                    print(f"  Frozen at joint state: {freeze1_q.tolist()}")
                elif press_count == 2:
                    # UNFREEZE — calibration data is locked in.
                    print("\n[yellow #2] UNFREEZE — resuming teleop")
                    print("  Move leader near follower's frozen pose first to avoid jerk.")
                    state.frozen = False
                    state.accumulate = False
                    state.status_text = "TELEOP (drive to second pose)"
                    rc.start_teleoperation()
                elif press_count == 3:
                    # FREEZE 2 — verification snapshot only; no aruco accumulation.
                    print("\n[yellow #3] FREEZE 2 (verify) — stopping teleop")
                    rc.stop_teleoperation()
                    time.sleep(0.1)
                    state.frozen = True
                    state.accumulate = False  # do NOT mix freeze-2 frames into solve
                    state.status_text = "FROZEN 2 — snapshot only (no pooling)"
                    freeze2_q, freeze2_dir = _capture_freeze("freeze2")
                    print(f"  Frozen at joint state: {freeze2_q.tolist()}")
                elif press_count == 4:
                    # UNFREEZE second time — teleop to a safe end pose.
                    print("\n[yellow #4] UNFREEZE — resuming teleop")
                    print("  Move leader near follower's frozen pose first to avoid jerk.")
                    state.frozen = False
                    state.accumulate = False
                    state.status_text = "TELEOP (drive somewhere safe to end)"
                    rc.start_teleoperation()
                elif press_count == 5:
                    # END.
                    print("\n[yellow #5] END — solving + returning home")
                    rc.stop_teleoperation()
                    state.frozen = True
                    state.accumulate = False
                    break
            time.sleep(0.05)

        # --- Multi-frame solve from latest freeze accumulation
        state.stop = True
        time.sleep(0.2)
        obj_pool, img_pool, n_used, mean_resid = history.top_k_concatenated(top_k)
        if obj_pool is None or n_used == 0:
            print("\n  x no valid wrist detections accumulated -- skipping save")
        else:
            link_cfg = YAM_BASE_ONLY_CONFIG.links["larger_coarse_board"]
            board_length = link_cfg.board_length
            T_base_aruco = link_to_aruco_transform(link_cfg)
            T_cam_aruco, resid = _solve_multiframe_pose(
                obj_pool, img_pool, K_w, dist_w, board_length,
            )
            if T_cam_aruco is None:
                print("  x multi-frame solvePnP failed -- skipping save")
            elif freeze1_q is None:
                print("  x freeze 1 was never recorded -- can't run FK; skipping save")
            else:
                # FK at the FREEZE-1 joint state (where the aruco frames in the
                # pool were captured). Using live_q at yellow #5 is wrong since
                # the arm has been teleop'd through freeze 2 and elsewhere.
                solve_q = np.asarray(freeze1_q, dtype=np.float64)
                T_base_ee = _fk_base_to_ee(solve_q)
                T_ee_cam = (
                    np.linalg.inv(T_base_ee)
                    @ T_base_aruco
                    @ np.linalg.inv(T_cam_aruco)
                )
                print(f"  T_cam_aruco resid = {resid:.2f} px over {n_used} pooled frame(s)")
                print(f"  solve_q (frozen)  = {solve_q}")
                print(f"  T_ee_cam translation = {T_ee_cam[:3, 3]} (m, in ee/grasp_site frame)")
                out_path = Path(output_file).expanduser()
                _save_calibration_result(
                    output_file=out_path,
                    wrist_cam_name=wrist_cam_name,
                    arm=arm,
                    K=K_w, dist=dist_w,
                    T_ee_cam=T_ee_cam,
                    resolution=wrist_resolution,
                    n_used=n_used,
                    mean_resid_px=resid,
                    joint_state=list(solve_q),
                )

        # --- Route current -> intermediate -> home
        print("\n[cleanup] current -> INTERMEDIATE waypoint")
        _smooth_move(follower, intermediate_pose, max_vel_rad_s=max_joint_vel_rad_s)
        print("[cleanup] intermediate -> HOME")
        _smooth_move(follower, HOME_POSE, max_vel_rad_s=max_joint_vel_rad_s)

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
        try:
            rc.cleanup()
        except SystemExit:
            pass


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("right", "left"), default="right")
    p.add_argument("--scene_cam", default="scene_camera")
    p.add_argument("--wrist_cam", default="",
                   help="Wrist camera name. Empty -> {arm}_wrist_camera.")
    p.add_argument("--camera_config_file",
                   default=str(Path.home() / ".config" / "raiden" / "camera.json"))
    p.add_argument("--output_file", default="",
                   help="Output JSON. Empty -> ~/.config/raiden/wrist_calibration_result.json.")
    p.add_argument("--vis_port", type=int, default=9092)
    p.add_argument("--scene_resolution", default="HD1080",
                   choices=("HD720", "HD1080", "HD2K"))
    p.add_argument("--wrist_resolution", default="HD1080",
                   choices=("HD720", "HD1080", "HD2K"))
    p.add_argument("--viz_width", type=int, default=1280)
    p.add_argument("--jpeg_quality", type=int, default=75)
    p.add_argument("--viz_hz", type=int, default=10)
    p.add_argument("--max_joint_vel_rad_s", type=float, default=0.25)
    p.add_argument("--history_size", type=int, default=60)
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()

    wrist_cam = args.wrist_cam or f"{args.arm}_wrist_camera"
    output_file = args.output_file or str(
        _DEFAULT_OUT_DIR / "wrist_calibration_result.json"
    )
    run_wrist_calibrate(
        arm=args.arm,
        scene_cam_name=args.scene_cam,
        wrist_cam_name=wrist_cam,
        camera_config_file=args.camera_config_file,
        output_file=output_file,
        vis_port=args.vis_port,
        scene_resolution=args.scene_resolution,
        wrist_resolution=args.wrist_resolution,
        viz_width=args.viz_width,
        jpeg_quality=args.jpeg_quality,
        viz_hz=args.viz_hz,
        max_joint_vel_rad_s=args.max_joint_vel_rad_s,
        history_size=args.history_size,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
