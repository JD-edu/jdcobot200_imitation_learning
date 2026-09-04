#!/usr/bin/env python3
"""Convert JDCobot200 synthetic NPZ episodes to a LeRobot dataset.

Each dataset row contains the robot state and RGB image at time ``t`` and the
measured joint state at ``t+1`` as its action. LeRobot's ``delta_timestamps``
API assembles these standard one-step actions into ``t+1 ... t+N`` chunks when
the dataset is loaded for training.

Normally use ``create_lerobot_dataset.py`` to run generation, validation, and
conversion together. This standalone converter remains useful when changing
camera settings without rerunning IK.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

# MuJoCo reads this before it is imported. EGL supports rendering without X11.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "lerobot_conversion_hf"))

import mujoco
import numpy as np
try:
    # LeRobot 0.4.x does not re-export this class from lerobot.datasets.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as error:
    raise ImportError(
        "LeRobot is required. Install it with: "
        "python3 -m pip install 'lerobot>=0.4'"
    ) from error

from jdcobot200_mink_solver import SO101MinkSolver


ROOT = Path(__file__).resolve().parent
JOINT_NAMES = [*SO101MinkSolver.ARM_JOINT_NAMES, "gripper"]
LOCAL_XML = ROOT / "scene.xml"
# This exercise directory currently omits the included jdcobot200.xml, while
# the immediately preceding dataset exercise contains the identical scene and
# model. Prefer a self-contained local copy whenever one is added.
DEFAULT_XML = (
    LOCAL_XML
    if (ROOT / "jdcobot200.xml").exists()
    else ROOT.parent / "10_create_virtual_dataset" / "scene.xml"
)


def require_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model does not contain {kind.name} '{name}'")
    return int(object_id)


def robot_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    joint_names = [*SO101MinkSolver.ARM_JOINT_NAMES, "gripper_left"]
    joint_ids = [
        require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    return model.jnt_qposadr[np.asarray(joint_ids, dtype=int)].astype(int)


def make_next_actions(states: np.ndarray) -> np.ndarray:
    """Return the measured joint state at t+1 (last state at episode end)."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or len(states) == 0:
        raise ValueError("states must be a non-empty [frames, joints] array")
    return np.concatenate((states[1:], states[-1:]), axis=0)


def make_camera(
    azimuth: float, elevation: float, distance: float, lookat: list[float]
) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    camera.lookat[:] = lookat
    return camera


def lerobot_features(height: int, width: int) -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
        "observation.images.front": {
            "dtype": "video",
            # LeRobot stores/decodes images as CHW. add_frame also accepts the
            # uint8 HWC array returned by MuJoCo and converts it internally.
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "synthetic_dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "lerobot_dataset")
    parser.add_argument(
        "--repo-id", default="local/jdcobot200_contact_pick_place",
        help="LeRobot dataset identifier (local output does not require upload)",
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--frame-stride", type=int, default=5,
        help="Keep every Nth 50 Hz IK frame (default 5 produces 10 Hz videos)",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--video-backend", default="pyav", choices=("pyav", "torchcodec"),
        help="LeRobot video decoder (pyav avoids local TorchCodec ABI issues)",
    )
    parser.add_argument(
        "--image-writer-threads", type=int, default=8,
        help="Parallel PNG writer threads used while building video episodes",
    )
    parser.add_argument("--azimuth", type=float, default=150.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--distance", type=float, default=0.9)
    parser.add_argument("--lookat", type=float, nargs=3, default=[0.15, 0.0, 0.15])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.horizon < 1
        or args.width < 1
        or args.height < 1
        or args.image_writer_threads < 0
        or args.frame_stride < 1
    ):
        raise ValueError(
            "horizon, width, and height must be positive; "
            "image-writer-threads cannot be negative"
        )
    manifest_path = args.input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run generate_synthetic_trajectories.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_fps = int(manifest["control_hz"])
    if source_fps % args.frame_stride:
        raise ValueError("control_hz must be divisible by frame-stride")
    fps = source_fps // args.frame_stride
    episode_files = sorted(args.input_dir.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {args.input_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite to replace it"
            )
        import shutil
        shutil.rmtree(args.output_dir)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        robot_type="jdcobot200",
        fps=fps,
        features=lerobot_features(args.height, args.width),
        use_videos=True,
        video_backend=args.video_backend,
        image_writer_threads=args.image_writer_threads,
    )

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    qpos_addresses = robot_qpos_addresses(model)
    front_camera = make_camera(
        args.azimuth, args.elevation, args.distance, list(args.lookat)
    )
    wrist_camera_id = require_id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera"
    )
    wrist_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(wrist_camera)
    wrist_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    wrist_camera.fixedcamid = wrist_camera_id
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        for episode_number, episode_file in enumerate(episode_files):
            with np.load(episode_file) as episode:
                qpos = np.asarray(episode["qpos"])[::args.frame_stride]
                states = qpos[:, qpos_addresses].astype(np.float32)
            actions = make_next_actions(states)

            for frame_index in range(len(states)):
                data.qpos[:] = qpos[frame_index]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=front_camera)
                front_image = renderer.render().copy()
                renderer.update_scene(data, camera=wrist_camera)
                wrist_image = renderer.render().copy()
                dataset.add_frame({
                    "observation.state": states[frame_index],
                    "observation.images.front": front_image,
                    "observation.images.wrist": wrist_image,
                    "action": actions[frame_index],
                    "task": "Pick the red block and place it on the green target",
                })
            dataset.save_episode()
            print(
                f"[{episode_number + 1}/{len(episode_files)}] "
                f"{episode_file.name}: {len(states)} frames"
            )
    finally:
        renderer.close()

    dataset.finalize()
    # Reopen through the public reader exactly as training code does. This also
    # verifies that t+1..t+horizon is returned as one action chunk and that
    # LeRobot supplies the episode-boundary padding mask.
    delta_timestamps = {
        # action[t] is state[t+1], therefore offsets 0..N-1 correspond to
        # measured states t+1..t+N.
        "action": [step / fps for step in range(args.horizon)]
    }
    loaded = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.output_dir,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    sample = loaded[0]
    expected = (args.horizon, len(JOINT_NAMES))
    if tuple(sample["action"].shape) != expected:
        raise RuntimeError(
            f"Reloaded action shape is {tuple(sample['action'].shape)}, "
            f"expected {expected}"
        )
    print(f"Reload verification passed: action shape={expected}")
    print(f"Saved LeRobot dataset to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
