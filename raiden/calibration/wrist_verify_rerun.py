"""Live wrist + scene extrinsics verification in Rerun.

Reads the saved wrist_calibration_result.json and raiden's calibration_results.json
(scene cam), runs teleop, and streams five panels to a Rerun web viewer:

  scene/rgb_aruco    scene cam + aruco markers + board outline (sanity)
  scene/rgb_overlay  scene cam + mujoco mesh overlay + red silhouette
                     (verifies T_base_scene_cam + the robot/exo mesh placement)
  wrist/rgb_overlay  wrist cam + mujoco mesh overlay + red silhouette
                     (verifies T_ee_wrist_cam: as you teleop and the wrist
                      looks at the YAM base, the rendered base+exo mesh should
                      track the real base in the image)
  wrist/rgb_dots     wrist cam with a uniform pixel grid drawn as FILLED red dots
  scene/rgb_dots     scene cam with OPEN red circles at the same physical floor
                     points (z=0 plane back-projection from the wrist grid).
                     Filled-dot-i in wrist <-> open-circle-i in scene point to
                     the same floor location if both extrinsics are correct.

Press the YELLOW leader button to end + return home.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mujoco
import numpy as np

# Self-locate from this file's path so the script works from any sshfs mount.
# Lives at raiden_fork/raiden/calibration/wrist_verify_rerun.py.
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
    _render_with_intrinsics,
    _compose_overlay,
    _PoseHistory,
    _solve_multiframe_pose,
)


ARM = "right"
SCENE_CAM_NAME = "scene_camera"
WRIST_CAM_NAME = "right_wrist_camera"
CAM_CONFIG_PATH = Path.home() / ".config" / "raiden" / "camera.json"
SCENE_CAL_PATH = Path.home() / ".config" / "raiden" / "calibration_results.json"
WRIST_CAL_PATH = Path.home() / ".config" / "raiden" / "wrist_calibration_result.json"

GRID_N = 5  # 5x5 = 25 candidate dots; filtered to valid
VIZ_WIDTH = 1280
JPEG_QUALITY = 75
VIZ_HZ = 10
INTERMEDIATE_POSE_RIGHT = np.array(
    [0.0097, 1.9213, 1.4582, -0.4324, 0.5880, -0.1459, 0.9923], dtype=np.float64
)
HOME_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class _State:
    stop: bool = False


def _load_T(path: Path, cam_name: str) -> np.ndarray:
    with open(path) as f:
        c = json.load(f)
    e = c["cameras"][cam_name]["extrinsics"]
    T = np.eye(4)
    T[:3, :3] = np.array(e["rotation_matrix"])
    T[:3, 3] = np.array(e["translation_vector"])
    return T


def _live_calibrate_scene_cam_in_base(scene_cam, K_s, dist_s, exo_cfg,
                                      n_frames: int = 30, top_k: int = 10):
    """Solve scene-cam pose in arm-base frame from a short live aruco capture.

    Replaces reading the (stale) extrinsic from calibration_results.json. Same
    multi-frame pooling + corner-origin shift convention exo_calibrate uses.
    Returns T_cam_in_base (4x4); raises if not enough detections.
    """
    from ExoConfigs.exoskeleton import link_to_aruco_transform
    from exo_utils import do_est_aruco_pose, ARUCO_DICT
    link_cfg = exo_cfg.links["larger_coarse_board"]
    aruco_board = exo_cfg.aruco_board_objects["larger_coarse_board"]
    board_length = link_cfg.board_length
    T_link_to_aruco = link_to_aruco_transform(link_cfg)
    T_aruco_to_link = np.linalg.inv(T_link_to_aruco)

    # Total expected corners on the board (each marker contributes 4 corners).
    try:
        total_corners = int(len(aruco_board.getIds())) * 4
    except Exception:
        total_corners = 36  # 3x3 board fallback

    history = _PoseHistory(max_size=n_frames)
    grabbed = 0
    attempts = 0
    pct_samples = []
    t0 = time.monotonic()
    print(f"[live-recal scene] collecting {n_frames} frames "
          f"(board: larger_coarse_board, L={board_length:.3f} m, "
          f"{total_corners} corners expected)…")
    while grabbed < n_frames and (time.monotonic() - t0) < 10.0:
        bgr = _grab_bgr(scene_cam)
        if bgr is None:
            time.sleep(0.01); continue
        attempts += 1
        try:
            r = do_est_aruco_pose(bgr, ARUCO_DICT, aruco_board, board_length,
                                  cameraMatrix=K_s, distCoeffs=dist_s)
        except Exception:
            r = -1
        if r == -1:
            print(f"  attempt {attempts:>3}: 0/{total_corners} corners "
                  f"( 0.0%) — no markers detected")
            continue
        # Recover corner-origin obj_pts (exact convention exo_calibrate uses).
        rvec, tvec_shifted = r["rtvec"]
        R_mat = cv2.Rodrigues(rvec)[0]
        center_offset_board = np.array(
            [board_length / 2, board_length / 2, 0], dtype=np.float64)
        tvec_corner = tvec_shifted - R_mat.dot(center_offset_board)
        obj_cam, img_pts = r["obj_img_pts"]
        n_detected = int(len(img_pts))
        pct = 100.0 * n_detected / max(total_corners, 1)
        pct_samples.append(pct)
        obj_board = (R_mat.T @ (obj_cam.T - tvec_corner.reshape(3, 1))).T
        proj_chk, _ = cv2.projectPoints(obj_board.astype(np.float32),
                                          rvec, tvec_corner, K_s, dist_s)
        resid = float(np.linalg.norm(
            proj_chk.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1).mean())
        print(f"  attempt {attempts:>3}: {n_detected}/{total_corners} corners "
              f"({pct:5.1f}%) resid={resid:.2f}px")
        T_aruco_in_cam_single = r["est_aruco_pose"]
        history.add(obj_board, img_pts, resid,
                    single_frame_t=T_aruco_in_cam_single[:3, 3])
        grabbed += 1

    obj_pool, img_pool, n_used, mean_resid = history.top_k_concatenated(k=top_k)
    if obj_pool is None or len(obj_pool) < 4:
        raise RuntimeError(
            f"live scene-cam recal: only {grabbed}/{attempts} aruco "
            f"detections, aim the scene cam at the exo board")
    T_aruco_in_cam, mf_resid = _solve_multiframe_pose(
        obj_pool, img_pool, K_s, dist_s, board_length)
    if T_aruco_in_cam is None:
        raise RuntimeError("live scene-cam recal: multi-frame solve failed")
    T_link_in_cam = T_aruco_in_cam @ T_aruco_to_link
    T_cam_in_base = np.linalg.inv(T_link_in_cam)
    mean_pct = float(np.mean(pct_samples)) if pct_samples else 0.0
    min_pct = float(np.min(pct_samples)) if pct_samples else 0.0
    max_pct = float(np.max(pct_samples)) if pct_samples else 0.0
    print(f"[live-recal scene] OK  used={n_used} frames, "
          f"corners detected: mean={mean_pct:.1f}%, min={min_pct:.1f}%, "
          f"max={max_pct:.1f}% (of {total_corners})")
    print(f"[live-recal scene] avg single-frame resid={mean_resid:.2f}px, "
          f"multi-frame={mf_resid:.2f}px")
    print(f"[live-recal scene] cam_in_base xyz="
          f"({T_cam_in_base[0,3]:+.3f}, {T_cam_in_base[1,3]:+.3f}, {T_cam_in_base[2,3]:+.3f})")
    if mean_pct < 70.0:
        print(f"  ⚠ WARNING: only {mean_pct:.0f}% of corners visible on "
              f"average — consider repositioning the scene cam so the "
              f"whole exo board is in view")
    return T_cam_in_base


def _fk_base_to_ee(joint_state_7: np.ndarray) -> np.ndarray:
    from i2rt.robots.kinematics import Kinematics
    from raiden._xml_paths import get_yam_4310_linear_xml_path
    global _KIN_CACHE
    try:
        kin = _KIN_CACHE
    except NameError:
        kin = Kinematics(get_yam_4310_linear_xml_path(), site_name="grasp_site")
        globals()["_KIN_CACHE"] = kin
    q = np.zeros(8, dtype=np.float64)
    q[:6] = joint_state_7[:6]
    return np.asarray(kin.fk(q), dtype=np.float64)


def _smooth_move(robot, target_7, max_vel=0.25, min_s=0.5, steps=100):
    from raiden.robot.controller import smooth_move_joints
    current = np.asarray(robot.get_joint_pos(), dtype=np.float64)
    target = np.asarray(target_7, dtype=np.float64)
    delta = float(np.max(np.abs(target - current)))
    t_s = max(min_s, delta / max(max_vel, 1e-6))
    print(f"    smooth_move dmax={delta:.3f} -> {t_s:.2f}s")
    smooth_move_joints(robot, target, time_interval_s=t_s, steps=steps)


def _capture_loop(scene_cam, wrist_cam, K_s, dist_s, W_s, H_s,
                  K_w, dist_w, W_w, H_w,
                  follower, T_base_scene, T_ee_wrist,
                  model, data, arm_qpos_idx, grip_qpos_idx,
                  exo_cfg, rr, state):
    from ExoConfigs.exoskeleton import link_to_aruco_transform
    from exo_utils import do_est_aruco_pose, ARUCO_DICT
    link_cfg = exo_cfg.links["larger_coarse_board"]
    aruco_board = exo_cfg.aruco_board_objects["larger_coarse_board"]
    board_length = link_cfg.board_length
    T_base_aruco = link_to_aruco_transform(link_cfg)

    log_interval = 1.0 / max(VIZ_HZ, 1)
    last_log = 0.0
    frame_idx = 0
    while not state.stop:
        scene_bgr = _grab_bgr(scene_cam)
        wrist_bgr = _grab_bgr(wrist_cam)
        if scene_bgr is None or wrist_bgr is None:
            time.sleep(0.01)
            continue

        # --- Update mujoco qpos from live follower joints
        try:
            live_q = follower.get_joint_pos()
            for idx, q in zip(arm_qpos_idx, live_q[:len(arm_qpos_idx)]):
                data.qpos[idx] = float(q)
            if grip_qpos_idx and len(live_q) >= 7:
                cmd = float(np.clip(live_q[6], 0.0, 1.0))
                for qidx, sign, stroke in grip_qpos_idx:
                    data.qpos[qidx] = sign * cmd * stroke
            mujoco.mj_forward(model, data)
        except Exception:
            live_q = None

        if live_q is None:
            time.sleep(0.01)
            continue

        T_base_ee = _fk_base_to_ee(np.asarray(live_q))
        T_base_wrist = T_base_ee @ T_ee_wrist
        T_link_in_scenecam = np.linalg.inv(T_base_scene)
        T_link_in_wristcam = np.linalg.inv(T_base_wrist)

        # --- scene/rgb_aruco
        scene_ann = scene_bgr.copy()
        try:
            r = do_est_aruco_pose(scene_bgr, ARUCO_DICT, aruco_board, board_length,
                                  cameraMatrix=K_s, distCoeffs=dist_s)
            if r != -1:
                scene_ann = r["pose_vis"]
                _draw_aruco_plane(scene_ann, r["est_aruco_pose"], board_length, K_s)
        except Exception:
            pass

        # --- scene/rgb_overlay + wrist/rgb_overlay
        # Render mujoco scene from each cam pose, alpha-blend, add red contours.
        overlay_s_bgr = scene_bgr.copy()
        overlay_w_bgr = wrist_bgr.copy()
        try:
            rendered_s, seg_s = _render_with_intrinsics(
                model, data, T_link_in_scenecam, K_s, W_s, H_s
            )
            comp_s = _compose_overlay(scene_bgr, rendered_s, seg_mask=seg_s)
            overlay_s_bgr = cv2.cvtColor(comp_s, cv2.COLOR_RGB2BGR)
            fg_s = (seg_s[:, :, 0] != -1).astype(np.uint8) * 255
            cs, _ = cv2.findContours(fg_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_s_bgr, cs, -1, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception as e:
            if not getattr(_capture_loop, "_scene_err", False):
                import traceback
                print("[verify] scene render failed:")
                traceback.print_exc()
                _capture_loop._scene_err = True

        try:
            rendered_w, seg_w = _render_with_intrinsics(
                model, data, T_link_in_wristcam, K_w, W_w, H_w
            )
            comp_w = _compose_overlay(wrist_bgr, rendered_w, seg_mask=seg_w)
            overlay_w_bgr = cv2.cvtColor(comp_w, cv2.COLOR_RGB2BGR)
            fg_w = (seg_w[:, :, 0] != -1).astype(np.uint8) * 255
            cw, _ = cv2.findContours(fg_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_w_bgr, cw, -1, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception as e:
            if not getattr(_capture_loop, "_wrist_err", False):
                import traceback
                print("[verify] wrist render failed:")
                traceback.print_exc()
                _capture_loop._wrist_err = True

        # --- Dot grid (uniform on wrist image -> z=0 in base -> project to scene)
        pad = 0.15
        us = np.linspace(W_w * pad, W_w * (1 - pad), GRID_N)
        vs = np.linspace(H_w * pad, H_w * (1 - pad), GRID_N)
        uu, vv = np.meshgrid(us, vs)
        grid_uv = np.stack([uu.flatten(), vv.flatten()], axis=-1)

        fx_w, fy_w = K_w[0, 0], K_w[1, 1]
        cx_w, cy_w = K_w[0, 2], K_w[1, 2]
        O = T_base_wrist[:3, 3]
        R_bw = T_base_wrist[:3, :3]
        valid_uv = []
        pts_3d = []
        for u, v in grid_uv:
            d_cam = np.array([(u - cx_w) / fx_w, (v - cy_w) / fy_w, 1.0])
            d_base = R_bw @ d_cam
            if abs(d_base[2]) < 1e-6:
                continue
            t = -O[2] / d_base[2]
            if t <= 0:
                continue
            valid_uv.append([u, v])
            pts_3d.append(O + t * d_base)
        valid_uv = np.array(valid_uv) if valid_uv else np.zeros((0, 2))
        pts_3d = np.array(pts_3d) if pts_3d else np.zeros((0, 3))

        wrist_dots = wrist_bgr.copy()
        scene_dots = scene_bgr.copy()
        red = (0, 0, 255)
        gray = (140, 140, 140)
        n_in_scene = 0
        if len(pts_3d):
            T_scene_base = np.linalg.inv(T_base_scene)
            P_h = np.c_[pts_3d, np.ones(len(pts_3d))]
            P_sc = (T_scene_base @ P_h.T).T[:, :3]
            front = P_sc[:, 2] > 0.05
            z_safe = np.where(front, P_sc[:, 2], 1.0)
            us_s = K_s[0, 0] * P_sc[:, 0] / z_safe + K_s[0, 2]
            vs_s = K_s[1, 1] * P_sc[:, 1] / z_safe + K_s[1, 2]
            in_bounds = (front & (us_s >= 0) & (us_s < W_s)
                         & (vs_s >= 0) & (vs_s < H_s))
            n_in_scene = int(in_bounds.sum())
            for i, ((u, v), ok) in enumerate(zip(valid_uv, in_bounds)):
                color = red if ok else gray
                cv2.circle(wrist_dots, (int(round(u)), int(round(v))),
                           10, color, -1, cv2.LINE_AA)
                cv2.putText(wrist_dots, str(i),
                            (int(round(u)) + 12, int(round(v)) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(wrist_dots, str(i),
                            (int(round(u)) + 12, int(round(v)) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                if ok:
                    cv2.circle(scene_dots,
                               (int(round(us_s[i])), int(round(vs_s[i]))),
                               14, red, 3, cv2.LINE_AA)
                    cv2.putText(scene_dots, str(i),
                                (int(round(us_s[i])) + 14, int(round(vs_s[i])) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(scene_dots, str(i),
                                (int(round(us_s[i])) + 14, int(round(vs_s[i])) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2, cv2.LINE_AA)

        # --- Log
        now = time.monotonic()
        if now - last_log >= log_interval:
            last_log = now
            import rerun as _rr
            rr.set_time("step", sequence=frame_idx)
            rr.log("scene/rgb_aruco",
                   _to_rerun_jpeg(scene_ann, target_w=VIZ_WIDTH, quality=JPEG_QUALITY))
            rr.log("scene/rgb_overlay",
                   _to_rerun_jpeg(overlay_s_bgr, target_w=VIZ_WIDTH, quality=JPEG_QUALITY))
            rr.log("wrist/rgb_overlay",
                   _to_rerun_jpeg(overlay_w_bgr, target_w=VIZ_WIDTH, quality=JPEG_QUALITY))
            rr.log("wrist/rgb_dots",
                   _to_rerun_jpeg(wrist_dots, target_w=VIZ_WIDTH, quality=JPEG_QUALITY))
            rr.log("scene/rgb_dots",
                   _to_rerun_jpeg(scene_dots, target_w=VIZ_WIDTH, quality=JPEG_QUALITY))
            rr.log("info", _rr.TextDocument(
                f"verify | dots: {len(pts_3d)}/{GRID_N**2} hit z=0, "
                f"{n_in_scene} project into scene | live_q={live_q}"
            ))
        frame_idx += 1


def main():
    import rerun as rr
    from raiden.robot.controller import RobotController
    from ExoConfigs.yam_exo import YAM_BASE_ONLY_CONFIG
    from exo_utils import get_link_poses_from_robot, position_exoskeleton_meshes

    # --- Wrist extrinsic from saved hand-eye calibration (ee-fixed; safe to cache).
    T_ee_wrist = _load_T(WRIST_CAL_PATH, WRIST_CAM_NAME)
    print(f"T_ee_wrist translation: {T_ee_wrist[:3, 3]}")

    # --- Cameras
    with open(CAM_CONFIG_PATH) as f:
        cam_cfg = json.load(f)
    scene_serial = int(cam_cfg[SCENE_CAM_NAME]["serial"])
    wrist_serial = int(cam_cfg[WRIST_CAM_NAME]["serial"])
    scene_cam, K_s, dist_s, W_s, H_s = _open_zed_native(scene_serial, "HD1080")
    wrist_cam, K_w, dist_w, W_w, H_w = _open_zed_native(wrist_serial, "HD1080")

    # --- Scene-cam extrinsic: solve live from aruco at startup (never trust
    # the saved one — camera gets bumped between sessions).
    from ExoConfigs.yam_exo import YAM_BASE_ONLY_CONFIG as _exo_cfg_for_recal
    T_base_scene = _live_calibrate_scene_cam_in_base(
        scene_cam, K_s, dist_s, _exo_cfg_for_recal)
    print(f"T_base_scene translation: {T_base_scene[:3, 3]}")

    # --- Mujoco model (mirror exo_calibrate's setup)
    os.chdir(str(_EXO_DIR))
    exo_cfg = YAM_BASE_ONLY_CONFIG
    xml_str = exo_cfg.xml.replace(
        '<compiler angle="radian"',
        '<compiler angle="radian" balanceinertia="true"',
    )
    xml_str = re.sub(r'<include file="[^"]*background[^"]*"\s*/>', '', xml_str)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    link_poses = get_link_poses_from_robot(exo_cfg, model, data)
    position_exoskeleton_meshes(exo_cfg, model, data, link_poses)
    print(f"  built mujoco model ({model.nbody} bodies)")

    arm_qpos_idx, grip_qpos_idx = [], []
    for jn in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            arm_qpos_idx.append(int(model.jnt_qposadr[jid]))
    for jn in ("left_finger", "right_finger", "joint7", "joint8"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            rng = model.jnt_range[jid]
            sign = +1.0 if abs(rng[1]) >= abs(rng[0]) else -1.0
            max_stroke = float(max(abs(rng[0]), abs(rng[1])))
            grip_qpos_idx.append((int(model.jnt_qposadr[jid]), sign, max_stroke))
    print(f"  qpos arm={arm_qpos_idx}, gripper={grip_qpos_idx}")

    # --- Rerun
    _init_rerun(9092)

    # --- Robot
    rc = RobotController(
        use_right_leader=(ARM == "right"),
        use_left_leader=(ARM == "left"),
        use_right_follower=(ARM == "right"),
        use_left_follower=(ARM == "left"),
    )
    rc.setup_for_teleop_recording()
    rc.enable_estop()
    rc.start_teleoperation()
    follower = rc.follower_r if ARM == "right" else rc.follower_l

    state = _State()
    th = threading.Thread(
        target=_capture_loop, daemon=True,
        args=(scene_cam, wrist_cam, K_s, dist_s, W_s, H_s,
              K_w, dist_w, W_w, H_w,
              follower, T_base_scene, T_ee_wrist,
              model, data, arm_qpos_idx, grip_qpos_idx,
              exo_cfg, rr, state),
    )
    th.start()

    print()
    print("=" * 72)
    print("  Teleop freely. Watch the 5 rerun panels:")
    print("    scene/rgb_aruco    scene cam + aruco")
    print("    scene/rgb_overlay  scene cam + mesh (verifies scene extrinsic)")
    print("    wrist/rgb_overlay  wrist cam + mesh (verifies wrist extrinsic)")
    print("    wrist/rgb_dots     wrist cam + filled red grid")
    print("    scene/rgb_dots     scene cam + open red circles (back-projected)")
    print("  Press YELLOW leader button to end + return home.")
    print("=" * 72)
    print()

    try:
        while True:
            if rc.check_button_press() is not None:
                print("\n[yellow] END — returning home")
                rc.stop_teleoperation()
                break
            time.sleep(0.05)

        state.stop = True
        time.sleep(0.2)
        print("[cleanup] current -> INTERMEDIATE")
        _smooth_move(follower, INTERMEDIATE_POSE_RIGHT)
        print("[cleanup] intermediate -> HOME")
        _smooth_move(follower, HOME_POSE)
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


if __name__ == "__main__":
    main()
