"""Exoskeleton-based scene-camera calibration for the YAM bimanual rig.

Detects the ArUco board glued onto the 3D-printed exoskeleton mounted on the
robot arm's base (yam_base_board_v2). From the detection we recover the
robot-base frame in camera coordinates; inverting gives ``T_cam_in_robot_base``,
which is what raiden's ``calibration_results.json`` stores for scene cameras.

Visualization runs as a Rerun web viewer (browser-friendly, no X server needed)
with these streams:
  - ``scene/rgb_aruco``: live frame with ArUco markers + axes + board outline
  - ``scene/rgb_overlay``: same frame composited with the full mujoco render of
    the robot mesh + green exoskeleton mesh + textured ArUco plane at the
    recovered camera pose
  - ``scene/cam_in_base_{x,y,z}``: scalar timeline of recovered camera position
  - ``info``: text panel with the latest detection + stability stats

Keyboard: Y → save the most recent valid detection, Q → quit without saving.
Stdin runs on the main thread; a daemon thread streams frames to Rerun.

By default the ZED scene camera is opened at HD1080 (1920x1080) — needed for
the 3x3 board to detect reliably. HD720 detection of the small board fails.

Usage:
    rd exo_calibrate                                # right arm, HD1080, scene_camera
    rd exo_calibrate --arm right --resolution HD1080
    rd exo_calibrate --resolution HD2K              # higher res; may fail PnP
"""
import os
# MUJOCO_GL must be set before any module that imports mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import quote

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

# Wire exo_redo into the import path (lives at raiden_fork/third_party/exo_redo)
_RAIDEN_ROOT = Path(__file__).resolve().parents[2]
_EXO_DIR = _RAIDEN_ROOT / "third_party" / "exo_redo"
if str(_EXO_DIR) not in sys.path:
    sys.path.insert(0, str(_EXO_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESOLUTION_MAP = {
    "HD720": None,    # filled at runtime from pyzed.sl
    "HD1080": None,
    "HD2K": None,
}

# Single shared renderer — creating two (one normal + one segmentation) hits an
# EGL context limit per process on some setups, causing the second Renderer's
# __init__ to fail silently and spew destructor stack-traces every frame.
# Use one Renderer and toggle enable_/disable_segmentation_rendering() instead.
_CACHED_RENDERER = None
_CACHED_RENDER_SHAPE = (0, 0)


def _scene_serial_from_config(camera_config_file: str, scene_cam_name: str) -> int:
    with open(camera_config_file) as f:
        cfg = json.load(f)
    if scene_cam_name not in cfg:
        raise RuntimeError(
            f"Camera {scene_cam_name!r} not in {camera_config_file}. "
            f"Available: {list(cfg)}"
        )
    return int(cfg[scene_cam_name]["serial"])


def _open_zed_native(serial: int, resolution: str):
    """Open the ZED at the chosen resolution via pyzed (NOT raiden's ZedCamera,
    which is hardcoded to HD720). Returns (cam, K, dist, W, H)."""
    import pyzed.sl as sl
    res_map = {
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD2K": sl.RESOLUTION.HD2K,
    }
    if resolution not in res_map:
        raise ValueError(f"resolution must be one of {list(res_map)}, got {resolution!r}")

    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(serial)
    init.camera_resolution = res_map[resolution]
    # HD1080+HD2K only support 15 fps max with depth NONE; HD720 supports 30.
    init.camera_fps = 30 if resolution == "HD720" else 15
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED (serial {serial}, {resolution}): {status}")

    info = cam.get_camera_information()
    cal = info.camera_configuration.calibration_parameters.left_cam
    W = info.camera_configuration.resolution.width
    H = info.camera_configuration.resolution.height
    K = np.array([[cal.fx, 0, cal.cx], [0, cal.fy, cal.cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(
        [cal.disto[0], cal.disto[1], cal.disto[2], cal.disto[3], cal.disto[4]],
        dtype=np.float64,
    )
    print(f"  ✓ Opened ZED serial={serial} at {resolution} → {W}x{H}, "
          f"K diag=({cal.fx:.1f},{cal.fy:.1f})")
    return cam, K, dist, W, H


def _grab_bgr(cam) -> Optional[np.ndarray]:
    import pyzed.sl as sl
    image = sl.Mat()
    rt = sl.RuntimeParameters()
    if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
        return None
    cam.retrieve_image(image, sl.VIEW.LEFT)
    return cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)


def _save_calibration(
    scene_cam_name: str,
    K: np.ndarray,
    dist: np.ndarray,
    T_cam_in_base: np.ndarray,
    calibration_path: Path,
    arm_label: str,
    resolution: str,
) -> None:
    cal: dict = {}
    if calibration_path.exists():
        with open(calibration_path) as f:
            cal = json.load(f)

    cal.setdefault("version", "1.0")
    cal["timestamp"] = datetime.now().isoformat(timespec="seconds")
    cal.setdefault("coordinate_frame", "left_arm_base")
    cal.setdefault("cameras", {})
    cal["cameras"][scene_cam_name] = {
        "type": "exoskeleton",
        "method": "yam_base_board_v2_aruco_3x3",
        "arm": arm_label,
        "resolution": resolution,
        "intrinsics": {
            "camera_matrix": K.tolist(),
            "distortion_coeffs": [float(v) for v in np.asarray(dist).reshape(-1)[:5]],
        },
        "num_poses_used": 1,
        "extrinsics": {
            "success": True,
            "rotation_matrix": T_cam_in_base[:3, :3].tolist(),
            "translation_vector": T_cam_in_base[:3, 3].tolist(),
            "reference_frame": f"{arm_label}_arm_base",
        },
    }

    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calibration_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\n  ✓ Saved calibration → {calibration_path}")
    print(f"      cam_in_base xyz = {T_cam_in_base[:3, 3]}")
    print(f"      reference_frame = {arm_label}_arm_base")


def _draw_aruco_plane(img, T_aruco_in_cam, board_length, K, color=(0, 255, 255)):
    L = board_length / 2.0
    corners = np.array([[-L, -L, 0], [+L, -L, 0], [+L, +L, 0], [-L, +L, 0]], dtype=np.float64)
    pts_cam = (T_aruco_in_cam[:3, :3] @ corners.T + T_aruco_in_cam[:3, 3:4]).T
    if (pts_cam[:, 2] <= 1e-3).any():
        return img
    proj = (K @ (pts_cam / pts_cam[:, 2:3]).T).T[:, :2].astype(np.int32)
    cv2.polylines(img, [proj.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    return img


def _compose_overlay(rgb_bgr: np.ndarray, rendered_rgb: np.ndarray,
                     seg_mask: Optional[np.ndarray] = None,
                     alpha: float = 0.7) -> np.ndarray:
    """Alpha-blend the mujoco render over the camera frame.

    If seg_mask is given (mujoco segmentation render, geom_id channel = -1 for
    background), use it for the foreground mask — this correctly preserves
    BLACK pixels in foreground geometry (e.g. ArUco marker squares). Falling
    back to a pixel-sum heuristic mis-classifies black aruco markers as
    background and creates a 'transparent black' artifact.
    """
    rgb_frame_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rendered_f = rendered_rgb.astype(np.float32)
    if seg_mask is not None:
        fg = (seg_mask[:, :, 0] != -1).astype(np.float32)[..., None]
    else:
        fg = (rendered_f.sum(axis=-1) >= 30).astype(np.float32)[..., None]
    composite = fg * (alpha * rendered_f + (1 - alpha) * rgb_frame_rgb) + (1 - fg) * rgb_frame_rgb
    return composite.clip(0, 255).astype(np.uint8)


def _render_with_intrinsics(model, data, T_link_in_cam, K, W, H):
    """Render the mujoco scene at the detected camera pose for overlay on
    a live camera frame (any K, any (cx, cy), any aspect).

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  IMPORTANT — DO NOT TRY ``cam_intrinsic`` HERE                       ║
    ║                                                                       ║
    ║  MuJoCo's ``cam_fovy`` renderer ASSUMES principal point = (W/2, H/2). ║
    ║  Real cameras (ZED especially) have an off-center (cx, cy). The diff ║
    ║  is a CONSTANT additive pixel shift of (cx − W/2, cy − H/2) on every  ║
    ║  rendered geom — exactly cv2.projectPoints' principal-point term.    ║
    ║                                                                       ║
    ║  CANONICAL FIX (this function):                                       ║
    ║      1. Render with cam_fovy only (no cam_intrinsic).                 ║
    ║      2. cv2.warpAffine the RGB + seg by (cx − W/2, cy − H/2).         ║
    ║      3. Composite with the seg-mask foreground (NOT pixel.sum()),     ║
    ║         so black aruco markers stay opaque.                           ║
    ║                                                                       ║
    ║  This works for SQUARE and NON-SQUARE pixel cameras alike, any K.    ║
    ║                                                                       ║
    ║  DO NOT set model.cam_intrinsic + cam_sensorsize programmatically —  ║
    ║  it doesn't engage cleanly through the Python API and breaks the     ║
    ║  projection by ~200 px (verified 2026-05-29; cost hours).             ║
    ║                                                                       ║
    ║  See memory: feedback-mujoco-principal-point-warp.                    ║
    ╚══════════════════════════════════════════════════════════════════════╝

    Returns ``(rgb, seg)`` — both shifted into the live-frame pixel grid,
    with edges filled with black RGB / -1 seg sentinel.
    """
    cam_id = model.cam('estimated_camera').id
    data.cam_xpos[cam_id] = np.linalg.inv(T_link_in_cam)[:3, 3]
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    data.cam_xmat[cam_id] = (flip @ T_link_in_cam[:3, :3]).T.reshape(-1)
    model.cam_fovy[cam_id] = float(np.degrees(2 * np.arctan(H / (2 * K[1, 1]))))

    # Single shared renderer — toggle segmentation mode between the two renders.
    # IMPORTANT: pre-created in the main thread (mujoco's EGL context is tied to
    # the thread that created it; trying to create a Renderer in the capture
    # daemon thread fails repeatedly with broken EGL state and spams
    # AttributeError: '_mjr_context' destructor noise every frame).
    global _CACHED_RENDERER, _CACHED_RENDER_SHAPE
    if _CACHED_RENDER_SHAPE != (W, H) or _CACHED_RENDERER is None:
        _CACHED_RENDERER = mujoco.Renderer(model, height=H, width=W)
        _CACHED_RENDER_SHAPE = (W, H)
    r = _CACHED_RENDERER
    # Normal RGB render
    r.disable_segmentation_rendering()
    r.update_scene(data, camera=cam_id)
    rgb = r.render()
    # Segmentation render (geom_id channel = -1 for background)
    r.enable_segmentation_rendering()
    r.update_scene(data, camera=cam_id)
    seg = r.render()
    r.disable_segmentation_rendering()

    # Shift to compensate for the off-center principal point in the live K.
    # mujoco renders with (cx_mj, cy_mj) = (W/2, H/2); real (cx, cy) may differ.
    dx = int(round(K[0, 2] - W / 2))   # +→ right
    dy = int(round(K[1, 2] - H / 2))   # +→ down
    if dx or dy:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        rgb = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_NEAREST,
                             borderValue=(0, 0, 0))
        # Shift the segmentation similarly; pad with -1 (background sentinel)
        seg_geom = seg[:, :, 0].astype(np.int32)
        seg_geom_shifted = cv2.warpAffine(
            seg_geom, M, (W, H), flags=cv2.INTER_NEAREST,
            borderValue=-1,
        )
        seg = np.stack([seg_geom_shifted, seg[:, :, 1]], axis=-1)
    return rgb, seg


# ---------------------------------------------------------------------------
# Rerun setup
# ---------------------------------------------------------------------------

def _init_rerun(vis_port: int):
    import rerun as rr
    rr.init("exo_calibrate")
    grpc_port = vis_port + 1
    server_uri = rr.serve_grpc(grpc_port=grpc_port)
    rr.serve_web_viewer(web_port=vis_port, open_browser=False)
    viewer_url = f"http://localhost:{vis_port}?url={quote(server_uri, safe='')}"
    print()
    print("=" * 64)
    print(f"  Rerun viewer (local):  {viewer_url}")
    print(f"  SSH tunnel (mac):       "
          f"ssh -L {vis_port}:localhost:{vis_port} "
          f"-L {grpc_port}:localhost:{grpc_port} russet")
    print("=" * 64)
    print()
    return rr


# ---------------------------------------------------------------------------
# Capture / detect / log loop
# ---------------------------------------------------------------------------

class _PoseHistory:
    """Sliding-window queue of per-frame (obj_pts, img_pts) ArUco correspondences
    + per-frame quality, used to compute a multi-frame robust pose.

    Adding statistically independent observations of the SAME (static) board
    from the SAME (static) camera tightens the pose far more than any
    rotation-averaging or pixel-median scheme: just pool all the
    correspondences and solve once.

    Quality score: (n_markers descending, then residual ascending). We keep
    a sliding ``max_size`` window and concatenate the top-K when asked.

    Motion guard: if a fresh single-frame translation jumps > reset_mm vs.
    the median of the recent queue, blow the queue away — board/camera moved
    and the older correspondences are stale.
    """

    def __init__(self, max_size: int = 30, reset_mm: float = 50.0):
        self.max_size = max_size
        self.reset_mm = reset_mm
        self.entries: list = []
        self._lock = threading.Lock()

    def add(self, obj_pts: np.ndarray, img_pts: np.ndarray,
            residual_px: float, single_frame_t: np.ndarray) -> None:
        with self._lock:
            if len(self.entries) >= 5:
                recent = np.stack([e["single_t"] for e in self.entries[-10:]])
                median_t = np.median(recent, axis=0)
                if np.linalg.norm(single_frame_t - median_t) * 1000 > self.reset_mm:
                    self.entries = []  # board/camera moved; reset
            self.entries.append({
                "obj": obj_pts.astype(np.float32).copy(),
                "img": img_pts.astype(np.float32).copy(),
                "n_markers": int(len(obj_pts) // 4),
                "residual": float(residual_px),
                "single_t": np.asarray(single_frame_t, dtype=np.float64).copy(),
            })
            if len(self.entries) > self.max_size:
                self.entries = self.entries[-self.max_size:]

    def top_k_concatenated(self, k: int = 10):
        """Return (obj_concat, img_concat, used_count, mean_residual)."""
        with self._lock:
            if not self.entries:
                return None, None, 0, float("nan")
            sorted_e = sorted(self.entries,
                              key=lambda e: (-e["n_markers"], e["residual"]))
            top = sorted_e[:k]
            obj = np.concatenate([e["obj"] for e in top], axis=0)
            img = np.concatenate([e["img"] for e in top], axis=0)
            return obj, img, len(top), float(np.mean([e["residual"] for e in top]))

    def stats(self) -> str:
        with self._lock:
            if not self.entries:
                return "no detections"
            n = len(self.entries)
            avg_m = float(np.mean([e["n_markers"] for e in self.entries]))
            avg_r = float(np.mean([e["residual"] for e in self.entries]))
            return f"hist {n}/{self.max_size} frames, avg {avg_m:.1f} markers, avg resid {avg_r:.2f} px"


def _solve_multiframe_pose(obj_pts: np.ndarray, img_pts: np.ndarray,
                            K: np.ndarray, dist: np.ndarray,
                            board_length: float):
    """Same post-processing as ``do_est_aruco_pose`` (ambiguity resolution,
    center-offset shift, Y/Z column flip), but applied to a pre-pooled
    correspondence set across multiple frames.

    Returns (T_aruco_in_cam_4x4, residual_px) or (None, None) on failure.
    """
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj_pts.astype(np.float32), img_pts.astype(np.float32),
        K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or not rvecs:
        return None, None

    # Ambiguity disambiguation — same heuristic as do_est_aruco_pose.
    best_idx = 0
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        R_mat, _ = cv2.Rodrigues(rv)
        normal = R_mat @ np.array([0., 0., 1.])
        if tv[2] > 0 and normal[2] < 0:
            best_idx = i
            break
    rvec = rvecs[best_idx][:, 0]
    tvec = tvecs[best_idx][:, 0]

    R_mat = cv2.Rodrigues(rvec)[0]
    center_offset_board = np.array([board_length / 2, board_length / 2, 0],
                                    dtype=np.float64)
    tvec_shifted = tvec + R_mat.dot(center_offset_board)

    est = np.eye(4)
    est[:3, 3] = tvec_shifted
    est[:3, :3] = R_mat
    est[:, 1:-1] *= -1  # same Y/Z column flip do_est_aruco_pose applies

    # Reprojection residual (using pre-shift rvec/tvec on the original
    # corner-origin obj_pts that solvePnP was given).
    proj, _ = cv2.projectPoints(obj_pts.astype(np.float32), rvec, tvec, K, dist)
    residual = float(np.linalg.norm(
        proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1).mean())
    return est, residual


class _DetectionState:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self._rolling_pos = []
        self.stop_event = threading.Event()

    def update(self, K, dist, T_cam_in_base):
        with self._lock:
            self._latest = (K.copy(), dist, T_cam_in_base.copy())
            self._rolling_pos.append(T_cam_in_base[:3, 3].copy())
            if len(self._rolling_pos) > 30:
                self._rolling_pos.pop(0)

    def snapshot(self):
        with self._lock:
            if self._latest is None:
                return None
            K, d, T = self._latest
            return K.copy(), d, T.copy()

    def stability_str(self):
        with self._lock:
            if len(self._rolling_pos) < 5:
                return ""
            stdev = np.std(self._rolling_pos, axis=0) * 1000
            return f"std (mm): {stdev[0]:.1f}, {stdev[1]:.1f}, {stdev[2]:.1f}"


def _resize_for_viz(img_bgr: np.ndarray, target_w: int = 640) -> np.ndarray:
    if img_bgr.shape[1] <= target_w:
        return img_bgr
    h, w = img_bgr.shape[:2]
    new_h = int(round(h * target_w / w))
    return cv2.resize(img_bgr, (target_w, new_h), interpolation=cv2.INTER_AREA)


def _to_rerun_jpeg(img_bgr: np.ndarray, target_w: int = 640, quality: int = 75):
    """Resize → JPEG-encode → wrap as rr.EncodedImage. Sending compressed bytes
    over the gRPC pipe is ~50× lighter than raw HD1080 RGB, which prevents the
    rerun viewer backlog that otherwise lags the live stream by seconds."""
    import rerun as rr
    small = _resize_for_viz(img_bgr, target_w=target_w)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return rr.Image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    return rr.EncodedImage(contents=bytes(buf), media_type="image/jpeg")


def _capture_loop(cam, K, dist, W, H, exo_cfg, model, data, rr, state,
                  solve_intrinsics: bool = False,
                  history_size: int = 30, top_k: int = 10,
                  robot_controller=None, arm_qpos_idx=None, grip_qpos_idx=None,
                  viz_width: int = 1280, viz_hz: float = 10.0,
                  jpeg_quality: int = 75):
    from ExoConfigs.exoskeleton import link_to_aruco_transform
    from exo_utils import do_est_aruco_pose, ARUCO_DICT

    # mujoco's EGL context is thread-bound; create one explicitly in THIS
    # daemon thread before any Renderer.
    try:
        _gl_ctx = mujoco.GLContext(W, H)
        _gl_ctx.make_current()
        print(f"[exo_calibrate] daemon-thread GL context created ({W}x{H})")
    except Exception as _exc:
        print(f"[exo_calibrate] failed to create daemon GL context: {_exc}")
        _gl_ctx = None

    link_cfg = exo_cfg.links["larger_coarse_board"]
    aruco_board = exo_cfg.aruco_board_objects["larger_coarse_board"]
    board_length = link_cfg.board_length
    T_link_to_aruco = link_to_aruco_transform(link_cfg)
    T_aruco_to_link = np.linalg.inv(T_link_to_aruco)

    history = _PoseHistory(max_size=history_size)
    state.history = history  # expose to main thread for stats display
    state.pose_locked = False
    state.locked_T_aruco_in_cam = None
    print(f"[exo_calibrate] multi-frame robust pose: history={history_size}, "
          f"pool top-{top_k}; solve_intrinsics={solve_intrinsics}")
    if robot_controller is not None:
        print(f"[exo_calibrate] teleop enabled; press the YELLOW (top) leader "
              f"button to TOGGLE pose lock (freezes the aruco pose so you can "
              f"drive the arm and visually compare).")

    frame_idx = 0
    last_log_t = 0.0
    LOG_INTERVAL_S = 1.0 / max(viz_hz, 0.1)

    while not state.stop_event.is_set():
        bgr = _grab_bgr(cam)
        if bgr is None:
            time.sleep(0.005)
            continue

        try:
            # Per-frame: detect markers + solve single-frame pose.
            # When --solve_intrinsics, pass cameraMatrix=None so do_est_aruco_pose
            # jointly solves for focal length each frame (ZED's K is bypassed).
            result = do_est_aruco_pose(
                bgr, ARUCO_DICT, aruco_board, board_length,
                cameraMatrix=(None if solve_intrinsics else K),
                distCoeffs=dist,
            )
        except Exception as e:
            result = -1

        info_text = ""
        if result == -1:
            aruco_bgr = bgr.copy()
            cv2.putText(aruco_bgr, "no ArUco detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            overlay_bgr = aruco_bgr
            info_text = "no ArUco detected"
        else:
            # K used for THIS frame's solve — either ZED K or per-frame jointly-solved.
            K_this = np.asarray(result["cameraMatrix"], dtype=np.float64)
            dist_this = (np.asarray(result["distCoeffs"], dtype=np.float64).reshape(-1)
                         if result.get("distCoeffs") is not None else dist)

            # Single-frame pose (for jump-detection + axes overlay) and the raw
            # 2D-3D correspondences to feed the multi-frame solver.
            T_aruco_in_cam_single = result["est_aruco_pose"]
            # ── KEY GOTCHA ───────────────────────────────────────────────────
            # do_est_aruco_pose returns rtvec=(rvec, tvec_AFTER_center_shift)
            # but obj_img_pts holds obj_cam computed with tvec_BEFORE_shift
            # (i.e. the corner-origin convention solvePnP used). To recover
            # obj_pts in board CORNER frame (matching what we'll re-feed to
            # solvePnP via _solve_multiframe_pose), undo the center shift on
            # tvec FIRST, then back-project. Otherwise obj_board ends up in
            # CENTER frame and the multi-frame solver double-shifts → rendered
            # plane translates by (L/2, L/2, 0) in board X/Y. ◣ debugged the
            # hard way when render's top-left landed at board's center.
            rvec, tvec_shifted = result["rtvec"]
            R_mat = cv2.Rodrigues(rvec)[0]
            center_offset_board = np.array(
                [board_length / 2, board_length / 2, 0], dtype=np.float64)
            tvec_corner = tvec_shifted - R_mat.dot(center_offset_board)
            obj_cam, img_pts = result["obj_img_pts"]
            obj_board = (R_mat.T @ (obj_cam.T - tvec_corner.reshape(3, 1))).T
            # Per-frame residual (proxy for detection confidence) — project the
            # corner-origin obj_board with the corner-origin (rvec, tvec_corner).
            proj_chk, _ = cv2.projectPoints(obj_board.astype(np.float32),
                                              rvec, tvec_corner,
                                              K_this, dist_this)
            residual_px = float(np.linalg.norm(
                proj_chk.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1).mean())

            # Skip history updates after lock — frozen pose is sacred.
            if not state.pose_locked:
                history.add(obj_board, img_pts, residual_px,
                            single_frame_t=T_aruco_in_cam_single[:3, 3])

            # Pose:
            #  - When LOCKED: use the frozen pose from the moment of lock.
            #  - Otherwise: re-solve multi-frame from the top-K pooled correspondences.
            obj_pool, img_pool, n_used, mean_resid = history.top_k_concatenated(k=top_k)
            multi_resid = float("nan")
            if state.pose_locked and state.locked_T_aruco_in_cam is not None:
                T_aruco_in_cam = state.locked_T_aruco_in_cam
            else:
                T_aruco_in_cam = T_aruco_in_cam_single  # fallback
                if obj_pool is not None and len(obj_pool) >= 4:
                    mf, mf_resid = _solve_multiframe_pose(
                        obj_pool, img_pool, K_this, dist_this, board_length,
                    )
                    if mf is not None:
                        T_aruco_in_cam = mf
                        multi_resid = mf_resid

            # Yellow-button rising edge → toggle pose lock. On lock-down we
            # snapshot the current best pose so teleop can drive the arm
            # without the pose drifting under it.
            if robot_controller is not None:
                if robot_controller.check_button_press() is not None:
                    state.pose_locked = not state.pose_locked
                    if state.pose_locked:
                        state.locked_T_aruco_in_cam = T_aruco_in_cam.copy()
                        print(f"\n[exo_calibrate] POSE LOCKED — teleop your "
                              f"leader and watch the rendered arm follow. "
                              f"Yellow again to unlock.")
                    else:
                        state.locked_T_aruco_in_cam = None
                        print(f"\n[exo_calibrate] pose UNLOCKED — tracking resumed.")

            T_link_in_cam = T_aruco_in_cam @ T_aruco_to_link
            T_cam_in_base = np.linalg.inv(T_link_in_cam)
            state.update(K_this, dist_this, T_cam_in_base)

            # Drive the rendered arm's joints from the live follower proprio
            # (so user can move the leader and visually compare the rendered
            # arm to the real arm in the camera frame).
            if (robot_controller is not None and arm_qpos_idx is not None
                    and robot_controller.follower_r is not None):
                try:
                    follower_q = robot_controller.follower_r.get_joint_pos()
                    for idx, q in zip(arm_qpos_idx, follower_q[:len(arm_qpos_idx)]):
                        data.qpos[idx] = float(q)
                    if grip_qpos_idx and len(follower_q) >= 7:
                        # Normalized gripper command in [0, 1] from raiden's proprio.
                        cmd = float(np.clip(follower_q[6], 0.0, 1.0))
                        for qidx, sign, stroke in grip_qpos_idx:
                            data.qpos[qidx] = sign * cmd * stroke
                    mujoco.mj_forward(model, data)
                except Exception:
                    pass

            aruco_bgr = result["pose_vis"]
            aruco_bgr = _draw_aruco_plane(aruco_bgr, T_aruco_in_cam, board_length, K)

            # Full mujoco render of robot + exo + aruco-plane at the recovered cam pose.
            try:
                rendered_rgb, seg_mask = _render_with_intrinsics(model, data, T_link_in_cam, K, W, H)
                overlay_rgb = _compose_overlay(bgr, rendered_rgb, seg_mask=seg_mask)
                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                # Draw cv2-projected aruco corners as a green ground-truth overlay
                _draw_aruco_plane(overlay_bgr, T_aruco_in_cam, board_length, K,
                                  color=(0, 255, 0))
                # Red-contour outline of the rendered mesh, drawn ON TOP of the
                # alpha-blended overlay. Highlights the render silhouette so it is
                # easy to compare against the real robot.
                fg_mask = (seg_mask[:, :, 0] != -1).astype(np.uint8) * 255
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_bgr, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
            except Exception as e:
                overlay_bgr = aruco_bgr.copy()
                # Log once on first failure so we see the actual error in terminal
                if not getattr(_capture_loop, '_render_err_logged', False):
                    import traceback
                    print(f"\n[exo_calibrate] mujoco render failed:")
                    traceback.print_exc()
                    print(f"[exo_calibrate] falling back to ArUco-only overlay\n")
                    _capture_loop._render_err_logged = True

            t = T_cam_in_base[:3, 3]
            lock_str = "[LOCKED]" if state.pose_locked else "[tracking]"
            info_text = (
                f"{lock_str} cam_in_base xyz=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f})  "
                + state.stability_str()
                + f"  | multi-frame: n_used={n_used} mean_pool_resid={multi_resid:.2f}px  "
                + history.stats()
            )
            rr.set_time("step", sequence=frame_idx)
            rr.log("scene/cam_in_base_x", rr.Scalars(float(t[0])))
            rr.log("scene/cam_in_base_y", rr.Scalars(float(t[1])))
            rr.log("scene/cam_in_base_z", rr.Scalars(float(t[2])))

        now = time.monotonic()
        if now - last_log_t >= LOG_INTERVAL_S:
            last_log_t = now
            rr.set_time("step", sequence=frame_idx)
            # JPEG-encode + resize before logging so the gRPC stream stays caught
            # up with the live ZED. Raw HD1080 RGB at every frame backlogs the
            # rerun viewer by seconds; compressed/resized stays real-time.
            rr.log("scene/rgb_aruco",
                   _to_rerun_jpeg(aruco_bgr, target_w=viz_width, quality=jpeg_quality))
            rr.log("scene/rgb_overlay",
                   _to_rerun_jpeg(overlay_bgr, target_w=viz_width, quality=jpeg_quality))
            rr.log("info", rr.TextDocument(info_text))
        frame_idx += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_exo_calibrate(
    scene_cam_name: str = "scene_camera",
    camera_config_file: Optional[str] = None,
    calibration_out: Optional[str] = None,
    arm: str = "right",
    vis_port: int = 9092,
    resolution: str = "HD1080",
    solve_intrinsics: bool = False,
    history_size: int = 30,
    top_k: int = 10,
    teleop: bool = True,
    viz_width: int = 1280,
    viz_hz: float = 10.0,
    jpeg_quality: int = 75,
) -> None:
    from raiden._config import CAMERA_CONFIG, CALIBRATION_FILE
    from ExoConfigs.yam_exo import YAM_BASE_ONLY_CONFIG
    from exo_utils import get_link_poses_from_robot, position_exoskeleton_meshes

    camera_config_file = camera_config_file or CAMERA_CONFIG
    calibration_out = Path(calibration_out or CALIBRATION_FILE)

    # Build the mujoco model from the exo config (robot + exo + textured aruco plane).
    # cd into exo_redo so the XML's relative texture/mesh paths resolve.
    os.chdir(str(_EXO_DIR))
    exo_cfg = YAM_BASE_ONLY_CONFIG
    xml_str = exo_cfg.xml.replace(
        '<compiler angle="radian"',
        '<compiler angle="radian" balanceinertia="true"',
    )
    # Strip background include so the render shows only robot + exo + aruco plane
    # (gray background was muddying the alpha overlay).
    import re
    xml_str = re.sub(r'<include file="[^"]*background[^"]*"\s*/>', '', xml_str)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    link_poses = get_link_poses_from_robot(exo_cfg, model, data)
    position_exoskeleton_meshes(exo_cfg, model, data, link_poses)
    print(f"  ✓ Built mujoco model ({model.nbody} bodies)")

    serial = _scene_serial_from_config(camera_config_file, scene_cam_name)
    cam, K, dist, W, H = _open_zed_native(serial, resolution)

    # IMPORTANT: don't pre-create the mujoco renderer in main thread. mujoco's
    # EGL context is thread-bound; creating it here locks out the capture
    # daemon thread, where the actual rendering happens. The daemon will
    # create its own Renderer on first call via _CACHED_RENDERER logic.

    # Look up the YAM arm + gripper joint qpos indices by name, so we can
    # drive them from live follower proprio without assuming model layout
    # (handles both free-base and welded-base XMLs robustly).
    arm_qpos_idx, grip_qpos_idx = [], []
    for jn in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            arm_qpos_idx.append(int(model.jnt_qposadr[jid]))
    # Gripper-finger joint names vary across XMLs:
    #   - i2rt's bundled YAM model uses 'left_finger' / 'right_finger' with
    #     OPPOSITE-signed ranges and an equality `right = -left` constraint.
    #   - raiden's combined yam_4310_linear XML uses 'joint7' / 'joint8' with
    #     SAME-signed [0, 0.0475] ranges and an equality `right = +left`.
    # mj_forward doesn't honor equality constraints, so we have to drive both
    # finger qpos slots ourselves. Determine per-finger sign + stroke from the
    # joint range so the same code path works for both XMLs.
    #
    # grip_qpos_idx becomes a list of (qpos_idx, sign, max_stroke_m).
    for jn in ("left_finger", "right_finger", "joint7", "joint8"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            rng = model.jnt_range[jid]
            sign = +1.0 if abs(rng[1]) >= abs(rng[0]) else -1.0
            max_stroke = float(max(abs(rng[0]), abs(rng[1])))
            grip_qpos_idx.append((int(model.jnt_qposadr[jid]), sign, max_stroke))
    print(f"  ✓ Model qpos indices: arm={arm_qpos_idx}, gripper={grip_qpos_idx}")

    # Optional teleop: leader-follower mirroring on the right arm so the user
    # can drive the real robot AND the rendered arm from the same proprio.
    robot_controller = None
    if teleop:
        try:
            from raiden.robot.controller import RobotController
            print("\nInitialising teleop (right leader + right follower)...")
            robot_controller = RobotController(
                use_right_leader=(arm == "right"),
                use_left_leader=(arm == "left"),
                use_right_follower=(arm == "right"),
                use_left_follower=(arm == "left"),
            )
            # setup_for_teleop_recording() internally calls check_can_interfaces()
            # AND initialize_robots() — calling them explicitly first causes the
            # second init pass to fail with "fail to communicate with motor".
            robot_controller.setup_for_teleop_recording()
            robot_controller.enable_estop()
            robot_controller.start_teleoperation()
            print("  ✓ Teleop active — move the leader, the follower (and the "
                  "rendered arm) will track.")
        except Exception as e:
            print(f"  ⚠ Teleop init failed ({e}); continuing without it. "
                  "Use --no-teleop to suppress.")
            robot_controller = None

    rr = _init_rerun(vis_port)
    state = _DetectionState()
    thread = threading.Thread(
        target=_capture_loop,
        args=(cam, K, dist, W, H, exo_cfg, model, data, rr, state),
        kwargs=dict(solve_intrinsics=solve_intrinsics,
                    history_size=history_size, top_k=top_k,
                    robot_controller=robot_controller,
                    arm_qpos_idx=arm_qpos_idx, grip_qpos_idx=grip_qpos_idx,
                    viz_width=viz_width, viz_hz=viz_hz,
                    jpeg_quality=jpeg_quality),
        daemon=True, name="exo-capture",
    )
    thread.start()

    print("\nWatch the Rerun viewer in your browser, then return here.")
    print("  y → save the latest detection")
    print("  q → quit without saving\n")

    try:
        while True:
            try:
                ans = input("[y]=save / [q]=quit > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "q"
            if ans == "q":
                print("Quitting without saving.")
                break
            if ans == "y":
                snap = state.snapshot()
                if snap is None:
                    print("  no valid detection yet — point camera at board and retry")
                    continue
                K_s, dist_s, T_s = snap
                _save_calibration(scene_cam_name, K_s, dist_s, T_s, calibration_out, arm, resolution)
                break
            print(f"  unknown input {ans!r} — use 'y' or 'q'")
    finally:
        state.stop_event.set()
        thread.join(timeout=2.0)
        if robot_controller is not None:
            try:
                robot_controller.stop_teleoperation()
            except Exception:
                pass
            try:
                robot_controller.return_to_home()
            except Exception:
                pass
            try:
                robot_controller.shutdown()
            except Exception:
                pass
        try:
            cam.close()
        except Exception:
            pass


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scene_cam", default="scene_camera")
    p.add_argument("--camera_config", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--arm", default="right", choices=["left", "right"])
    p.add_argument("--vis_port", type=int, default=9092)
    p.add_argument("--resolution", default="HD1080", choices=["HD720", "HD1080", "HD2K"],
                   help="ZED resolution (default HD1080 — needed for the 3x3 board)")
    p.add_argument("--solve_intrinsics", action="store_true",
                   help="Per-frame jointly solve focal length with pose (cameraMatrix=None). "
                        "Default off — use ZED factory K. ZED's K is trustworthy; turn on only "
                        "when you suspect drift or want to compare.")
    p.add_argument("--history_size", type=int, default=30,
                   help="Sliding-window size of past frames kept for multi-frame pooled pose.")
    p.add_argument("--top_k", type=int, default=10,
                   help="Pool the top-K best frames (by n_markers desc, residual asc) for the "
                        "multi-frame solvePnP.")
    p.add_argument("--teleop", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable leader-follower teleop on the active arm. While teleop runs, "
                        "the rendered arm tracks the live follower joints. Yellow leader "
                        "button toggles pose-lock so you can drive without the aruco pose "
                        "drifting. Use --no-teleop to disable.")
    args = p.parse_args()
    run_exo_calibrate(
        scene_cam_name=args.scene_cam,
        camera_config_file=args.camera_config,
        calibration_out=args.out,
        arm=args.arm,
        vis_port=args.vis_port,
        resolution=args.resolution,
        solve_intrinsics=args.solve_intrinsics,
        history_size=args.history_size,
        top_k=args.top_k,
        teleop=args.teleop,
    )


if __name__ == "__main__":
    main()
