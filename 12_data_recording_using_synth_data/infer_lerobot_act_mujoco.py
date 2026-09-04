#!/usr/bin/env python3
"""Run a standard LeRobot ACT checkpoint closed-loop in JDCobot200 MuJoCo."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "lerobot_inference_hf"))

import mujoco
import mujoco.viewer
import numpy as np
import torch

from generate_synthetic_trajectories import DEFAULT_XML, OPEN, ROOT
from jdcobot200_mink_solver import SO101MinkSolver
from lerobot_act_compat import enable_act_only_imports

enable_act_only_imports()

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (  # noqa: E402
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "outputs" / "act_jdcobot200" / "checkpoints" / "last" / "pretrained_model",
    )
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "lerobot_dataset")
    parser.add_argument("--repo-id", default="local/jdcobot200_pick_place")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run MuJoCo without opening the interactive viewer",
    )
    parser.add_argument(
        "--no-realtime", action="store_true",
        help="Do not synchronize simulation steps to the dataset FPS",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        try:
            # is_available() can still be true when all GPUs are busy or in an
            # unavailable compute mode. Test a real allocation before loading
            # the much larger checkpoint.
            probe = torch.empty(1, device="cuda")
            del probe
            return torch.device("cuda")
        except (RuntimeError, torch.AcceleratorError) as error:
            print(f"CUDA is visible but unavailable; falling back to CPU: {error}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"Missing MuJoCo object: {name}")
    return int(value)


def make_policy_camera() -> mujoco.MjvCamera:
    """Camera fixed to the exact view used to create the training dataset."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 150.0, -25.0, 0.9
    camera.lookat[:] = [0.15, 0.0, 0.15]
    return camera


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    device = choose_device(args.device)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    image_feature = metadata.features["observation.images.camera"]
    _, height, width = image_feature["shape"]

    # LeRobot saves the training device in config.json. Loading directly would
    # make safetensors allocate on that old device before `.to(device)` runs.
    # Force a portable CPU load first, then move to the currently selected one.
    policy_config = PreTrainedConfig.from_pretrained(args.checkpoint)
    if not isinstance(policy_config, ACTConfig):
        raise TypeError(
            f"Checkpoint policy must be ACT, got {type(policy_config).__name__}"
        )
    policy_config.device = "cpu"
    policy_config.use_amp = False
    policy = ACTPolicy.from_pretrained(
        args.checkpoint, config=policy_config
    ).to(device).eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": str(device)}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    arm_names = SO101MinkSolver.ARM_JOINT_NAMES
    arm_joints = np.array([
        require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in arm_names
    ])
    arm_qadr = model.jnt_qposadr[arm_joints]
    gripper_joint = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_left")
    gripper_qadr = int(model.jnt_qposadr[gripper_joint])
    arm_act = np.array([
        require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in arm_names
    ])
    gripper_act = require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor")
    block_joint = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_freejoint")
    block_qadr = int(model.jnt_qposadr[block_joint])
    grasp_weld = require_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "red_block_grasp")

    rng = np.random.default_rng(args.seed)
    data.qpos[arm_qadr] = np.array([0.0, -0.2, 0.75, 0.15, 2.14746])
    data.qpos[gripper_qadr] = OPEN
    data.qpos[block_qadr:block_qadr + 7] = np.r_[
        rng.uniform([0.18, -0.13], [0.25, -0.02]), 0.016, 1.0, 0.0, 0.0, 0.0
    ]
    data.ctrl[arm_act] = data.qpos[arm_qadr]
    data.ctrl[gripper_act] = OPEN
    data.eq_active[grasp_weld] = 0
    mujoco.mj_forward(model, data)

    policy_camera = make_policy_camera()
    renderer = mujoco.Renderer(model, height=height, width=width)
    policy.reset()
    dt = 1.0 / metadata.fps
    substeps = max(1, round(dt / model.opt.timestep))

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        )
        # The viewer camera can be moved freely by the user. Policy inference
        # always uses policy_camera above, so mouse movement cannot change the
        # model input distribution.
        viewer.cam.azimuth = 150.0
        viewer.cam.elevation = -25.0
        viewer.cam.distance = 0.9
        viewer.cam.lookat[:] = [0.15, 0.0, 0.15]

    try:
        for step in range(args.steps):
            if viewer is not None and not viewer.is_running():
                print("MuJoCo viewer closed; stopping inference")
                break
            wall_start = time.perf_counter()
            renderer.update_scene(data, camera=policy_camera)
            image = renderer.render().copy()
            state = np.r_[data.qpos[arm_qadr], data.qpos[gripper_qadr]].astype(np.float32)
            observation = {
                "observation.state": torch.from_numpy(state),
                "observation.images.camera": (
                    torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                ),
            }
            processed = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = postprocessor(action).squeeze(0).numpy()
            data.ctrl[arm_act] = action[:5]
            data.ctrl[gripper_act] = action[5]
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if step % 50 == 0:
                block = np.round(data.qpos[block_qadr:block_qadr + 3], 3)
                print(f"step={step:04d} action={np.round(action, 3)} block={block}")
            if not args.no_realtime:
                remaining = dt - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        renderer.close()
        if viewer is not None:
            viewer.close()
    print("Inference rollout complete")


if __name__ == "__main__":
    main()
