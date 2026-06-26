# Raiden (cameronosmith fork)

> **Fork notes — calibration assets and scripts.** This fork
> (cameronosmith/raiden_fork) is the version used for YAM recording and
> inference work on the cmu-tri team. It adds a print-ready ArUco board, a
> 3D-print mount for attaching the board to the YAM base, and the
> calibration scripts that consume them. The rest of the README is
> upstream tri-ml/raiden documentation.
>
> ## Calibration — usage
>
> 1. **Print the board.** Print `calibration_assets/aruco_board.png` at the
>    marker size your `raiden/calibration/core.py` config expects (default
>    matches the ChArUco layout in `third_party/exo_redo/ExoConfigs/exoskeleton.py`).
> 2. **3D-print the mount.** Slice and print
>    `calibration_assets/yam_base_board_v2.stl`. Bolt the printed paper
>    board to it and clamp the assembly onto the YAM base plate so the
>    board sits flat on top.
> 3. **Scene-camera extrinsic** (board ↔ YAM base):
>    ```
>    python -m raiden.calibration.exo_calibrate --arm right
>    ```
> 4. **Wrist-camera hand-eye:**
>    ```
>    python -m raiden.calibration.wrist_calibrate --arm right
>    ```
> 5. **Verify wrist calibration** with a live rerun overlay:
>    ```
>    python -m raiden.calibration.wrist_verify_rerun --arm right
>    ```
>
> ## What this fork adds (vs upstream tri-ml/raiden)
>
> **Print + 3D-print assets** (`calibration_assets/`):
> - `aruco_board.png` — print-ready ChArUco board.
> - `yam_base_board_v2.stl` — 3D-print mount that attaches the printed
>   paper board to the YAM base. (Source STL also lives at
>   `third_party/exo_redo/so100_blender_testings/yam_base_board_v2.stl`.)
>
> **Calibration source** (`raiden/calibration/`):
> - `exo_calibrate.py` — scene-camera extrinsic via the YAM-base ArUco
>   board.
> - `wrist_calibrate.py` — hand-eye calibration for wrist cameras using
>   on-arm board observations.
> - `wrist_verify_rerun.py` — live rerun overlay to visually verify a
>   wrist-calibration result.
> - `capture_wrist_camera_pose_joint_state.py` — paired (joint, image)
>   capture helper used by hand-eye datasets.
>
> **Board geometry source** (`third_party/exo_redo/`):
> - `ExoConfigs/exoskeleton.py` and `ExoConfigs/yam_exo.py` — board
>   layout, marker IDs, mount geometry.
> - `so100_blender_testings/` — blender source files for the YAM-base
>   board, including the canonical `yam_base_board_v2.stl`.
> - `robot_models/board_imgs/` — alternate board PNGs (alignment board,
>   coarser layouts) used for development.

Raiden is an end-to-end data collection toolkit for YAM robot arms. It covers
the full pipeline from hardware setup to policy-ready datasets: camera
calibration, teleoperation, multi-camera recording, dataset conversion, and
visualization.

**[Documentation](https://tri-ml.github.io/raiden/)** · **[Get started](https://tri-ml.github.io/raiden/guide/)**

**Key features**

- **Flexible control** — leader-follower teleoperation or SpaceMouse end-effector control, in bimanual or single-arm configurations.
- **Manipulability-aware IK** — uses [PyRoki](https://github.com/chungmin99/pyroki) and [J-Parse](https://jparse-manip.github.io/) for smooth and singularity-aware control.
- **Multiple depth backends** — IR structured light (RealSense), ZED SDK stereo, TRI Stereo, and [Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo) for high-quality depth tailored to manipulation scenes.
- **Heterogeneous cameras** — mix ZED and Intel RealSense cameras freely in a single session, across scene and wrist roles.
- **Automated extrinsic calibration** — hand-eye calibration for wrist cameras and static extrinsic estimation for scene cameras via ChArUco boards.
- **Metadata console** — a terminal UI (`rd console`) for reviewing demonstrations, correcting success/failure labels, and managing tasks and teachers.
- **Policy-ready output** — converts recordings to a simple, flat file format with synchronized frames, per-frame extrinsics, and interpolated joint poses, ready to plug into policy training frameworks.

## Installation

See the **[Installation guide](https://tri-ml.github.io/raiden/guide/installation/)** for full instructions.

## Commands

| Command | Description |
|---|---|
| `rd list_devices` | List all connected cameras, arms, and SpaceMouse devices |
| `rd record_calibration_poses` | Record robot poses for camera calibration |
| `rd calibrate` | Calibrate cameras (hand-eye + scene extrinsics) |
| `rd teleop` | Teleoperate arms without recording |
| `rd record` | Record teleoperation demonstrations |
| `rd replay` | Replay recorded follower arm motion |
| `rd console` | Browse and correct demonstration metadata in a terminal UI |
| `rd convert` | Convert successful recordings to a structured dataset |
| `rd shardify` | Export converted episodes to WebDataset shards |
| `rd visualize` | Visualize a converted recording with Rerun |
| `rd serve` | Start the policy server for live inference |
| `rd make_ffs_onnx` | Export Fast Foundation Stereo model to ONNX / TensorRT engines |
| `rd make_tri_stereo_engine` | Compile TRI Stereo TensorRT engine from ONNX model |

Run `rd <command> --help` for all options.

## Roadmap

The following features are coming soon:

- **Fin-ray gripper support** — support for fin-ray compliant grippers, which conform to object shapes for robust and gentle grasping.
- **Policy training and inference** — built-in integration for policy training pipelines and closed-loop inference.
- **LeRobot format converter** — export converted episodes to the [LeRobot](https://github.com/huggingface/lerobot) dataset format for compatibility with the Hugging Face ecosystem.
- **Initial scene condition management** — set up and save named initial scene conditions in the console to enable reproducible, side-by-side comparison of multiple policies under identical starting states.

## Disclaimer

Raiden is research software provided **as-is**, without warranty of any kind. Operating robotic arms involves inherent physical risks. The authors and Toyota Research Institute accept **no liability** for any damage to property, equipment, or persons arising from the use of this software.

## Citation

```bibtex
@misc{raiden2026,
  title  = {{RAIDEN}: A Toolkit for Policy Learning with {YAM} Bimanual Robot Arms},
  author = {Iwase, Shun and Miller, Patrick and Yao, Jonathan and Jatavallabhula, {Krishna Murthy} and Zakharov, Sergey},
  year   = {2026},
}
```
