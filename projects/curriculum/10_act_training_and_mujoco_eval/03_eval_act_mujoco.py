#!/usr/bin/env python3
"""학습한 LeRobot ACT policy를 합성 데이터와 같은 MuJoCo 환경에서 평가합니다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import types

# 창이 없는 환경에서도 dataset camera를 렌더링할 수 있게 MuJoCo import 전에 설정한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

import lerobot

# LeRobot 0.4.4의 policies/__init__.py가 사용하지 않는 GR00T까지 import하는 문제를
# 학습 코드와 동일하게 우회한다.
if "lerobot.policies" not in sys.modules:
    policies_package = types.ModuleType("lerobot.policies")
    policies_package.__path__ = [str(Path(lerobot.__file__).resolve().parent / "policies")]
    policies_package.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = policies_package

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COURSE08 = PROJECT_ROOT / "projects/curriculum/08_synthetic_pick_and_place"
COURSE09 = PROJECT_ROOT / "projects/curriculum/09_act_dataset_generation"
for path in (COURSE08, COURSE09):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dataset_tools  # noqa: E402


EXPERT = dataset_tools.EXPERT
SCENE_PATH = dataset_tools.SCENE_PATH
CAMERA_KEY = "observation.images.dataset_camera"
PREPROCESSOR_NAME = "policy_preprocessor.json"
POSTPROCESSOR_NAME = "policy_postprocessor.json"


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 torch.cuda.is_available()이 False입니다.")
    return device


def load_policy(model_dir: Path, device: torch.device):
    required = {
        "config.json", "model.safetensors", PREPROCESSOR_NAME, POSTPROCESSOR_NAME,
    }
    missing = sorted(name for name in required if not (model_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"모델 파일이 없습니다: {missing} (경로: {model_dir})")

    config = PreTrainedConfig.from_pretrained(model_dir, local_files_only=True)
    if not isinstance(config, ACTConfig):
        raise ValueError(f"ACT checkpoint가 아닙니다: {type(config).__name__}")
    config.device = str(device)
    policy = ACTPolicy.from_pretrained(
        model_dir, config=config, local_files_only=True
    ).to(device)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        model_dir,
        config_filename=PREPROCESSOR_NAME,
        local_files_only=True,
        overrides={"device_processor": {"device": str(device)}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        model_dir,
        config_filename=POSTPROCESSOR_NAME,
        local_files_only=True,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    policy.eval()
    return policy, preprocessor, postprocessor


def quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(
        2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
    ))


def evaluate_episode(
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    *,
    rng: np.random.Generator,
    episode_index: int,
    width: int,
    height: int,
    max_steps: int,
    max_arm_step_rad: float,
    max_gripper_step: float,
    viewer_enabled: bool,
    camera_preview: bool,
    fps: int,
    viewer_backend: str = "mujoco",
    final_hold_seconds: float = 0.0,
) -> dict:
    # 학습 데이터와 똑같이 IK 검사를 통과한 pick/place 분포에서 장면을 뽑는다.
    start, target, pick_yaw, place_yaw, _phases, attempts = dataset_tools.sample_plan(
        rng, tolerance_mm=1.0, path_steps=201, max_attempts=100,
    )
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    arm_addresses = np.asarray([
        model.jnt_qposadr[model.joint(name).id] for name in EXPERT.JOINT_NAMES
    ])
    arm_joint_ids = np.asarray([model.joint(name).id for name in EXPERT.JOINT_NAMES])
    arm_lower = model.jnt_range[arm_joint_ids, 0]
    arm_upper = model.jnt_range[arm_joint_ids, 1]
    left_address = int(model.jnt_qposadr[model.joint("gripper_left").id])
    right_address = int(model.jnt_qposadr[model.joint("gripper_right").id])
    block_joint = model.joint("block_free").id
    block_address = int(model.jnt_qposadr[block_joint])
    block_dof = int(model.jnt_dofadr[block_joint])
    tcp_site = model.site("graspframe").id
    left_body = model.body("gripper_left_assembly").id
    right_body = model.body("gripper_right_assembly").id
    target_body = model.body("place_target").id
    model.body_pos[target_body] = [target[0], target[1], 0.001]
    model.body_quat[target_body] = [
        np.cos(place_yaw / 2.0), 0.0, 0.0, np.sin(place_yaw / 2.0)
    ]

    def set_block(position: np.ndarray, yaw: float) -> None:
        data.qpos[block_address:block_address + 3] = position
        data.qpos[block_address + 3:block_address + 7] = [
            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
        ]
        data.qvel[block_dof:block_dof + 6] = 0.0

    def jaw_yaw() -> float:
        vector = data.xpos[right_body] - data.xpos[left_body]
        return float(np.arctan2(vector[1], vector[0]))

    data.qpos[arm_addresses] = EXPERT.READY_Q
    data.qpos[left_address] = EXPERT.GRIPPER_OPEN
    data.qpos[right_address] = -EXPERT.GRIPPER_OPEN
    set_block(start, pick_yaw)
    mujoco.mj_forward(model, data)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    attached = False
    was_attached = False
    tcp_offset = np.zeros(3, dtype=np.float64)
    viewer = None
    if viewer_enabled:
        try:
            if viewer_backend == "glfw":
                # mujoco.Renderer의 offscreen GLFW context가 GLFW_VISIBLE=False
                # window hint를 남긴다. Hint는 다음 create_window에도 유지되므로
                # 명시적으로 초기화하지 않으면 3D viewer도 숨김 창으로 생성된다.
                EXPERT.glfw.default_window_hints()
                EXPERT.glfw.window_hint(EXPERT.glfw.VISIBLE, EXPERT.glfw.TRUE)
                viewer = EXPERT.Viewer(model)
            elif viewer_backend == "mujoco":
                from mujoco import viewer as mujoco_viewer

                viewer = mujoco_viewer.launch_passive(model, data)
                viewer.cam.lookat[:] = [0.08, 0.11, 0.15]
                viewer.cam.distance = 0.72
                viewer.cam.azimuth = 145.0
                viewer.cam.elevation = -28.0
            else:
                raise ValueError(f"지원하지 않는 viewer backend: {viewer_backend}")
            if viewer_backend == "glfw":
                # 첫 policy 추론이 끝나기 전에도 초기 장면을 즉시 표시한다.
                EXPERT.glfw.make_context_current(viewer.window)
                viewer.render(model, data, f"episode={episode_index} | loading first action...")
        except Exception:
            renderer.close()
            raise

    cv2 = None
    if camera_preview:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise RuntimeError("--show에는 opencv-python이 필요합니다.") from exc
        cv2 = cv2_module

    try:
        for step in range(max_steps):
            if viewer is not None:
                if viewer_backend == "glfw" and viewer.should_close():
                    raise KeyboardInterrupt
                if viewer_backend == "mujoco" and not viewer.is_running():
                    raise KeyboardInterrupt
            renderer.update_scene(data, camera="dataset_camera")
            image = renderer.render().copy()
            image_tensor = (
                torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
            )
            state = np.r_[data.qpos[arm_addresses], data.qpos[left_address]].astype(
                np.float32
            )
            observation = {
                CAMERA_KEY: image_tensor,
                "observation.state": torch.from_numpy(state),
            }
            processed = preprocessor(observation)
            with torch.inference_mode():
                action = postprocessor(policy.select_action(processed))
            command = np.asarray(action, dtype=np.float32).reshape(-1)
            if command.shape != (6,) or not np.isfinite(command).all():
                raise RuntimeError(
                    f"episode {episode_index}, step {step}: 잘못된 action {command}"
                )

            current_arm = data.qpos[arm_addresses].copy()
            requested_arm = np.clip(command[:5], arm_lower, arm_upper)
            arm_delta = np.clip(
                requested_arm - current_arm, -max_arm_step_rad, max_arm_step_rad
            )
            next_arm = current_arm + arm_delta
            current_gripper = float(data.qpos[left_address])
            requested_gripper = float(np.clip(
                command[5], EXPERT.GRIPPER_CLOSED, EXPERT.GRIPPER_OPEN
            ))
            next_gripper = current_gripper + float(np.clip(
                requested_gripper - current_gripper,
                -max_gripper_step,
                max_gripper_step,
            ))
            data.qpos[arm_addresses] = next_arm
            data.qpos[left_address] = next_gripper
            data.qpos[right_address] = -next_gripper
            mujoco.mj_forward(model, data)

            block_position = data.qpos[block_address:block_address + 3].copy()
            if not attached and next_gripper <= EXPERT.GRIPPER_CLOSED + 0.08:
                position_error = block_position - data.site_xpos[tcp_site]
                yaw_error = EXPERT.axis_error(jaw_yaw(), pick_yaw + np.pi / 2.0)
                if (
                    np.linalg.norm(position_error[:2]) <= EXPERT.MAX_GRASP_XY_ERROR_M
                    and abs(position_error[2] + 0.035) <= EXPERT.MAX_GRASP_Z_ERROR_M
                    and yaw_error <= EXPERT.MAX_GRASP_YAW_ERROR_RAD
                ):
                    tcp_offset = position_error
                    attached = True
                    was_attached = True
            if attached and next_gripper >= EXPERT.GRIPPER_OPEN - 0.08:
                attached = False
            if attached:
                set_block(data.site_xpos[tcp_site] + tcp_offset, jaw_yaw() - np.pi / 2.0)
                mujoco.mj_forward(model, data)

            if cv2 is not None:
                display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    display,
                    f"episode={episode_index} step={step + 1}/{max_steps} "
                    f"grasp={'yes' if attached else 'no'}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )
                cv2.imshow("ACT MuJoCo - dataset_camera (Q/ESC: stop)", display)
                key = cv2.waitKey(max(1, int(round(1000 / 20)))) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt
            if viewer is not None:
                if viewer_backend == "glfw":
                    # offscreen dataset-camera renderer가 자신의 context를 활성화하므로
                    # 3D viewer를 그리기 전에 GLFW window context를 다시 선택한다.
                    EXPERT.glfw.make_context_current(viewer.window)
                    viewer.render(
                        model, data,
                        f"episode={episode_index} step={step + 1}/{max_steps}",
                    )
                else:
                    viewer.sync()
                time.sleep(1.0 / fps)

        # 성공 여부와 관계없이 마지막 자세를 잠시 남겨 결과를 눈으로 확인할 수 있게 한다.
        if viewer is not None and viewer_backend == "glfw" and final_hold_seconds > 0:
            deadline = time.monotonic() + final_hold_seconds
            while time.monotonic() < deadline and not viewer.should_close():
                EXPERT.glfw.make_context_current(viewer.window)
                viewer.render(model, data, f"episode={episode_index} | final pose")
                time.sleep(1.0 / min(fps, 60))
    finally:
        renderer.close()
        if viewer is not None:
            viewer.close()
        if cv2 is not None:
            cv2.destroyAllWindows()

    final_position = data.qpos[block_address:block_address + 3].copy()
    final_yaw = quaternion_yaw(data.qpos[block_address + 3:block_address + 7])
    position_error_mm = float(np.linalg.norm(final_position - target) * 1000.0)
    yaw_error_deg = float(np.rad2deg(EXPERT.axis_error(final_yaw, place_yaw)))
    success = bool(
        was_attached and not attached
        and position_error_mm <= 5.0 and yaw_error_deg <= 5.0
    )
    return {
        "episode": episode_index,
        "success": success,
        "grasped": was_attached,
        "released": was_attached and not attached,
        "position_error_mm": position_error_mm,
        "yaw_error_deg": yaw_error_deg,
        "sampling_attempts": attempts,
        "pick_xyz_m": start.tolist(),
        "target_xyz_m": target.tolist(),
        "pick_yaw_rad": float(pick_yaw),
        "target_yaw_rad": float(place_yaw),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="pretrained_model 폴더")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-arm-step-deg", type=float, default=5.0,
                        help="비정상 예측으로부터 보호하는 frame당 관절 변화 제한")
    parser.add_argument("--max-gripper-step", type=float, default=0.10)
    parser.add_argument("--fps", type=int, default=20,
                        help="MuJoCo 창에서 추론 action을 실행하는 속도")
    parser.add_argument("--headless", action="store_true",
                        help="MuJoCo 창 없이 평가만 실행")
    parser.add_argument("--camera-preview", action="store_true",
                        help="ACT가 보는 dataset camera 영상도 별도 창에 표시")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes와 max-steps는 1 이상이어야 합니다.")
    if args.width < 16 or args.height < 16:
        raise ValueError("width와 height는 16 이상이어야 합니다.")
    if not 1 <= args.fps <= 240:
        raise ValueError("fps는 1~240 범위여야 합니다.")
    if args.max_arm_step_deg <= 0 or args.max_gripper_step <= 0:
        raise ValueError("action 변화 제한은 양수여야 합니다.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    model_dir = args.model.resolve()
    policy, preprocessor, postprocessor = load_policy(model_dir, device)
    expected_shape = policy.config.input_features[CAMERA_KEY].shape
    if tuple(expected_shape) != (3, args.height, args.width):
        raise ValueError(
            f"모델 image shape={expected_shape}, 실행 shape="
            f"(3, {args.height}, {args.width})가 다릅니다."
        )

    rng = np.random.default_rng(args.seed)
    results = []
    print(
        f"ACT MuJoCo 평가 시작: episodes={args.episodes}, device={device}, "
        f"image={args.width}x{args.height}, max_steps={args.max_steps}"
    )
    for episode in range(1, args.episodes + 1):
        print(f"episode {episode}: 무작위 장면 생성 후 ACT 추론을 시작합니다.")
        result = evaluate_episode(
            policy, preprocessor, postprocessor,
            rng=rng, episode_index=episode, width=args.width, height=args.height,
            max_steps=args.max_steps,
            max_arm_step_rad=np.deg2rad(args.max_arm_step_deg),
            max_gripper_step=args.max_gripper_step,
            viewer_enabled=not args.headless,
            camera_preview=args.camera_preview,
            fps=args.fps,
        )
        results.append(result)
        print(
            f"episode {episode}: {'성공' if result['success'] else '실패'}, "
            f"grasp={result['grasped']}, release={result['released']}, "
            f"position={result['position_error_mm']:.2f}mm, "
            f"yaw={result['yaw_error_deg']:.2f}deg"
        )

    successes = sum(item["success"] for item in results)
    summary = {
        "model": str(model_dir),
        "device": str(device),
        "seed": args.seed,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "results": results,
    }
    print(f"평가 완료: success={successes}/{args.episodes} ({100 * successes / args.episodes:.1f}%)")
    if args.result_json is not None:
        output = args.result_json.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(output)
        print(f"결과 저장: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("사용자가 평가를 중단했습니다.")
        raise SystemExit(130)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1)
