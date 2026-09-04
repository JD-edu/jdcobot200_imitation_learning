#!/usr/bin/env python3
"""Run a standard LeRobot ACT checkpoint closed-loop in JDCobot200 MuJoCo."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--repo-id", default="local/jdcobot200_contact_pick_place")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grasp-close-threshold", type=float, default=0.10)
    parser.add_argument("--grasp-confirm-steps", type=int, default=3)
    parser.add_argument("--release-open-threshold", type=float, default=0.25)
    parser.add_argument("--release-goal-radius", type=float, default=0.040)
    parser.add_argument("--release-max-height", type=float, default=0.060)
    parser.add_argument("--release-confirm-steps", type=int, default=3)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "inference_report.json")
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


def bilateral_block_contact(
    data: mujoco.MjData, block_geom: int, pad_geoms: tuple[int, int]
) -> tuple[bool, bool]:
    touching = [False, False]
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        for side, pad in enumerate(pad_geoms):
            if pair == {pad, block_geom}:
                touching[side] = True
    return touching[0], touching[1]


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
    image_feature = metadata.features["observation.images.front"]
    _, height, width = image_feature["shape"]
    wrist_shape = metadata.features["observation.images.wrist"]["shape"]
    if list(wrist_shape) != list(image_feature["shape"]):
        raise ValueError("front and wrist image shapes must match")

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
    tool_site = require_id(model, mujoco.mjtObj.mjOBJ_SITE, "graspframe")
    gripper_body = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_assembly")
    block_body = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    block_geom = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_block_geom")
    pad_geoms = tuple(
        require_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("left_finger_pad", "right_finger_pad")
    )

    rng = np.random.default_rng(args.seed)
    data.qpos[arm_qadr] = np.array([0.0, -0.2, 0.75, 0.15, 2.14746])
    data.qpos[gripper_qadr] = OPEN
    data.qpos[block_qadr:block_qadr + 7] = np.r_[
        rng.uniform([0.18, -0.13], [0.25, -0.02]), 0.016, 1.0, 0.0, 0.0, 0.0
    ]
    data.ctrl[arm_act] = data.qpos[arm_qadr]
    data.ctrl[gripper_act] = OPEN
    mujoco.mj_forward(model, data)

    front_camera = make_policy_camera()
    wrist_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(wrist_camera)
    wrist_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    wrist_camera.fixedcamid = require_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    renderer = mujoco.Renderer(model, height=height, width=width)
    policy.reset()
    grasped = False
    grasp_completed = False
    open_command_steps = 0
    bilateral_steps = 0
    bilateral_contact_frames = 0
    grip_local_offset: np.ndarray | None = None
    grasp_step = None
    release_step = None
    minimum_grasp_distance = float("inf")
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
            renderer.update_scene(data, camera=front_camera)
            front_image = renderer.render().copy()
            renderer.update_scene(data, camera=wrist_camera)
            wrist_image = renderer.render().copy()
            state = np.r_[data.qpos[arm_qadr], data.qpos[gripper_qadr]].astype(np.float32)
            observation = {
                "observation.state": torch.from_numpy(state),
                "observation.images.front": (
                    torch.from_numpy(front_image).permute(2, 0, 1).float() / 255.0
                ),
                "observation.images.wrist": (
                    torch.from_numpy(wrist_image).permute(2, 0, 1).float() / 255.0
                ),
            }
            processed = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = postprocessor(action).squeeze(0).numpy()
            grasp_distance = float(np.linalg.norm(
                data.site_xpos[tool_site] - data.xpos[block_body]
            ))
            minimum_grasp_distance = min(minimum_grasp_distance, grasp_distance)
            left_contact, right_contact = bilateral_block_contact(
                data, block_geom, pad_geoms
            )
            bilateral_now = left_contact and right_contact
            bilateral_contact_frames += int(bilateral_now)
            bilateral_steps = bilateral_steps + 1 if bilateral_now else 0
            if (
                not grasped
                and not grasp_completed
                and bilateral_steps >= args.grasp_confirm_steps
                and action[5] <= args.grasp_close_threshold
            ):
                grasped = True
                rotation = data.xmat[gripper_body].reshape(3, 3)
                grip_local_offset = rotation.T @ (
                    data.xpos[block_body] - data.xpos[gripper_body]
                )
                grasp_step = step
                print(f"bilateral_contact_grasp=ON step={step}")
            if grasped:
                block_xyz = data.xpos[block_body]
                goal_distance = float(np.linalg.norm(
                    block_xyz[:2] - np.array([0.22, 0.10])
                ))
                release_ready = (
                    goal_distance <= args.release_goal_radius
                    and block_xyz[2] <= args.release_max_height
                    and action[5] >= args.release_open_threshold
                )
                open_command_steps = open_command_steps + 1 if release_ready else 0
                if open_command_steps >= args.release_confirm_steps:
                    grasped = False
                    grip_local_offset = None
                    grasp_completed = True
                    release_step = step
                    print(
                        f"contact_grasp=OFF step={step} "
                        f"goal_distance={goal_distance:.4f} m"
                    )
            data.ctrl[arm_act] = action[:5]
            data.ctrl[gripper_act] = action[5]
            for _ in range(substeps):
                data.xfrc_applied[block_body, :3] = 0.0
                if grip_local_offset is not None:
                    rotation = data.xmat[gripper_body].reshape(3, 3)
                    desired = data.xpos[gripper_body] + rotation @ grip_local_offset
                    relative_velocity = (
                        data.cvel[block_body, 3:6] - data.cvel[gripper_body, 3:6]
                    )
                    force = (
                        180.0 * (desired - data.xpos[block_body])
                        - 4.0 * relative_velocity
                    )
                    norm = np.linalg.norm(force)
                    if norm > 3.0:
                        force *= 3.0 / norm
                    data.xfrc_applied[block_body, :3] = force
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
    final_block = data.qpos[block_qadr:block_qadr + 3].copy()
    goal_error = float(np.linalg.norm(final_block[:2] - np.array([0.22, 0.10])))
    report = {
        "seed": args.seed,
        "steps": args.steps,
        "headless": args.headless,
        "device": str(device),
        "final_block_xyz": final_block.tolist(),
        "goal_xy_error_m": goal_error,
        "minimum_grasp_distance_m": minimum_grasp_distance,
        "grasp_step": grasp_step,
        "release_step": release_step,
        "grasp_active_at_end": grasped,
        "grasp_completed": grasp_completed,
        "bilateral_contact_frames": bilateral_contact_frames,
        "uses_weld": False,
        "contact_conditioned_compliant_grip": True,
        "success": bool(goal_error <= 0.04 and final_block[2] <= 0.05),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Inference rollout complete; report={args.report.resolve()}")


if __name__ == "__main__":
    main()
