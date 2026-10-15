#!/usr/bin/env python3
"""학습한 SmolVLA를 데이터 생성과 같은 MuJoCo pick-and-place에서 추론합니다."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import secrets
import sys
import tempfile
import types

# MuJoCo가 import되기 전에 렌더링 backend를 하나로 통일한다. EGL renderer와 GLX/GLFW
# viewer를 한 프로세스에서 섞으면 X_GLXMakeCurrent BadAccess가 발생할 수 있다.
os.environ["MUJOCO_GL"] = "egl" if "--headless" in sys.argv else "glfw"

import draccus
import numpy as np
import torch

import lerobot

# LeRobot 0.4.4에서 policies/__init__.py가 사용하지 않는 GR00T까지 import하는 문제를
# 학습 코드와 동일하게 우회한다.
if "lerobot.policies" not in sys.modules:
    package = types.ModuleType("lerobot.policies")
    package.__path__ = [str(Path(lerobot.__file__).resolve().parent / "policies")]
    package.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = package

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
# 저장된 pipeline을 복원하기 전에 SmolVLA 전용 step을 registry에 등록한다.
from lerobot.policies.smolvla.processor_smolvla import SmolVLANewLineProcessor  # noqa: F401
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)


LESSON_DIR = Path(__file__).resolve().parent
ACT_EVALUATOR_PATH = LESSON_DIR.parent / "10_act_training_and_mujoco_eval/03_eval_act_mujoco.py"
CAMERA_KEY = "observation.images.dataset_camera"
PREPROCESSOR_NAME = "policy_preprocessor.json"
POSTPROCESSOR_NAME = "policy_postprocessor.json"
DEFAULT_TASK = "Pick the red block and place it on the green target."
# lesson 10 평가기는 예전 lesson 08에 있던 kinematic grasp 허용 오차를 참조한다.
# 현재 expert에서는 그 상수가 제거되었으므로 추론 환경의 블록/TCP 치수에 맞춰 이
# 스크립트가 직접 소유한다. Z 기준값 -0.035 m는 평가기 안의 TCP grasp 높이와 같다.
GRASP_XY_TOLERANCE_M = 0.025
GRASP_Z_TOLERANCE_M = 0.015
GRASP_YAW_TOLERANCE_RAD = np.deg2rad(20.0)


def load_mujoco_evaluator():
    """검증된 lesson 10의 장면 초기화, 안전 실행, 성공 판정을 재사용한다."""
    spec = importlib.util.spec_from_file_location("curriculum10_mujoco_eval", ACT_EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MuJoCo 평가기를 읽을 수 없습니다: {ACT_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR = load_mujoco_evaluator()
# 재사용 평가기의 구버전 symbol 의존성을 현재 스크립트에서 보완한다. setattr을 쓰는
# 이유는 lesson 08 파일을 수정하거나 전역 설치 상태에 의존하지 않기 위해서다.
EVALUATOR.EXPERT.MAX_GRASP_XY_ERROR_M = GRASP_XY_TOLERANCE_M
EVALUATOR.EXPERT.MAX_GRASP_Z_ERROR_M = GRASP_Z_TOLERANCE_M
EVALUATOR.EXPERT.MAX_GRASP_YAW_ERROR_RAD = GRASP_YAW_TOLERANCE_RAD


def load_compatible_config(model_dir: Path) -> SmolVLAConfig:
    """저장 config의 policy discriminator를 제거한 뒤 concrete config로 읽는다."""
    config_path = model_dir / "config.json"
    with config_path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    policy_type = raw.pop("type", "smolvla")
    if policy_type != "smolvla":
        raise ValueError(f"SmolVLA checkpoint가 아닙니다: type={policy_type!r}")
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as stream:
        json.dump(raw, stream)
        stream.flush()
        with draccus.config_type("json"):
            return draccus.parse(SmolVLAConfig, stream.name, args=[])


class TaskPreprocessor:
    """환경 관측에 SmolVLA의 언어 명령을 매 step 삽입한다."""

    def __init__(self, pipeline, task: str):
        self.pipeline = pipeline
        self.task = task

    def reset(self) -> None:
        self.pipeline.reset()

    def __call__(self, observation):
        observation = dict(observation)
        observation["task"] = self.task
        return self.pipeline(observation)


def load_policy(model_dir: Path, device: torch.device, task: str):
    required = {
        "config.json", "model.safetensors", PREPROCESSOR_NAME, POSTPROCESSOR_NAME,
    }
    missing = sorted(name for name in required if not (model_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"모델 파일이 없습니다: {missing} (경로: {model_dir})")

    config = load_compatible_config(model_dir)
    config.device = str(device)
    policy = SmolVLAPolicy.from_pretrained(
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
    return policy, TaskPreprocessor(preprocessor, task), postprocessor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="학습 결과의 pretrained_model 폴더")
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help="학습 데이터와 같은 영어 task 문장")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int,
                        help="생략하면 실행할 때마다 무작위 seed를 자동 생성")
    parser.add_argument("--max-arm-step-deg", type=float, default=5.0,
                        help="비정상 예측으로부터 보호하는 frame당 관절 변화 제한")
    parser.add_argument("--max-gripper-step", type=float, default=0.10)
    parser.add_argument("--fps", type=int, default=20,
                        help="GUI에서 action을 실행하는 속도")
    parser.add_argument("--final-hold-seconds", type=float, default=3.0,
                        help="GUI에서 에피소드 종료 후 마지막 자세를 유지할 시간")
    parser.add_argument("--headless", action="store_true",
                        help="MuJoCo 창 없이 평가하고 성공률만 출력")
    parser.add_argument("--camera-preview", action="store_true",
                        help="policy가 보는 dataset camera 영상을 별도 창에 표시")
    parser.add_argument("--result-json", type=Path,
                        help="에피소드별 결과와 성공률을 저장할 JSON 경로")
    args = parser.parse_args()

    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes와 max-steps는 1 이상이어야 합니다.")
    if args.width < 16 or args.height < 16:
        raise ValueError("width와 height는 16 이상이어야 합니다.")
    if not 1 <= args.fps <= 240:
        raise ValueError("fps는 1~240 범위여야 합니다.")
    if args.max_arm_step_deg <= 0 or args.max_gripper_step <= 0:
        raise ValueError("action 변화 제한은 양수여야 합니다.")
    if args.final_hold_seconds < 0:
        raise ValueError("final-hold-seconds는 0 이상이어야 합니다.")
    if not args.task.strip():
        raise ValueError("task 문장은 비어 있을 수 없습니다.")

    seed = args.seed if args.seed is not None else secrets.randbits(32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = EVALUATOR.choose_device(args.device)
    model_dir = args.model.resolve()
    policy, preprocessor, postprocessor = load_policy(model_dir, device, args.task)
    expected_shape = policy.config.input_features[CAMERA_KEY].shape
    if tuple(expected_shape) != (3, args.height, args.width):
        raise ValueError(
            f"모델 image shape={expected_shape}, 실행 shape="
            f"(3, {args.height}, {args.width})가 다릅니다."
        )

    rng = np.random.default_rng(seed)
    results = []
    mode = "headless" if args.headless else "GUI"
    print(
        f"SmolVLA MuJoCo 추론 시작: mode={mode}, episodes={args.episodes}, "
        f"device={device}, seed={seed}, image={args.width}x{args.height}, "
        f"max_steps={args.max_steps}"
    )
    for episode in range(1, args.episodes + 1):
        print(f"episode {episode}: 무작위 장면 생성 후 추론을 시작합니다.")
        result = EVALUATOR.evaluate_episode(
            policy, preprocessor, postprocessor,
            rng=rng, episode_index=episode, width=args.width, height=args.height,
            max_steps=args.max_steps,
            max_arm_step_rad=np.deg2rad(args.max_arm_step_deg),
            max_gripper_step=args.max_gripper_step,
            viewer_enabled=not args.headless,
            camera_preview=args.camera_preview,
            fps=args.fps,
            viewer_backend="glfw",
            final_hold_seconds=args.final_hold_seconds,
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
        "model": str(model_dir), "task": args.task, "mode": mode,
        "device": str(device), "seed": seed, "episodes": args.episodes,
        "successes": successes, "success_rate": successes / args.episodes,
        "results": results,
    }
    print(
        f"추론 완료: success={successes}/{args.episodes} "
        f"({100 * successes / args.episodes:.1f}%)"
    )
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
        print("사용자가 추론을 중단했습니다.")
        raise SystemExit(130)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1)
