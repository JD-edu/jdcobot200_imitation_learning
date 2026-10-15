#!/usr/bin/env python3
"""11의 NPZ 데이터를 SmolVLA가 읽는 LeRobot Dataset v3로 변환합니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


CAMERA_KEY = "observation.images.dataset_camera"
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
DEFAULT_TASK = "Pick the red block and place it on the green target."


def features(height: int, width: int) -> dict:
    return {
        CAMERA_KEY: {"dtype": "image", "shape": (height, width, 3),
                     "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "debug.block_pose": {"dtype": "float32", "shape": (4,),
                             "names": ["x", "y", "z", "yaw"]},
        "debug.tcp_position": {"dtype": "float32", "shape": (3,),
                               "names": ["x", "y", "z"]},
        "debug.phase": {"dtype": "int64", "shape": (1,), "names": ["phase"]},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
        "next.success": {"dtype": "bool", "shape": (1,), "names": ["success"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).parent / "data/train")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "data/lerobot_jdcobot200_smolvla")
    parser.add_argument("--repo-id", default="local/jdcobot200_smolvla_pnp")
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"출력 폴더가 이미 존재합니다: {args.output}")
    if "/" not in args.repo_id:
        raise ValueError("repo-id는 namespace/name 형식이어야 합니다.")

    with (args.source / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise ValueError("변환할 성공 episode가 없습니다.")
    fps = int(manifest["fps"])
    height, width = map(int, manifest["image_size"])
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, root=args.output,
        robot_type="jdcobot200_mujoco", features=features(height, width),
        use_videos=False,
    )

    total_frames = 0
    for order, item in enumerate(episodes, 1):
        index = int(item["episode_index"])
        path = args.source / "episodes" / f"episode_{index:06d}.npz"
        with np.load(path) as data:
            required = {"images", "observation_state", "action", "phase",
                        "tcp_position", "block_pose"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path.name}: 필드 누락 {sorted(missing)}")
            frame_count = len(data["images"])
            if frame_count == 0 or any(len(data[key]) != frame_count for key in required):
                raise ValueError(f"{path.name}: frame 길이가 일치하지 않습니다.")
            if data["images"].shape != (frame_count, height, width, 3):
                raise ValueError(f"{path.name}: image shape 오류 {data['images'].shape}")
            if data["observation_state"].shape != (frame_count, 6):
                raise ValueError(f"{path.name}: state shape 오류")
            if data["action"].shape != (frame_count, 6):
                raise ValueError(f"{path.name}: action shape 오류")

            # observation(t)에 다음 제어 명령을 대응시킨다. 마지막 명령은 반복한다.
            next_actions = np.concatenate([data["action"][1:], data["action"][-1:]])
            for frame in range(frame_count):
                last = frame == frame_count - 1
                dataset.add_frame({
                    CAMERA_KEY: data["images"][frame],
                    "observation.state": data["observation_state"][frame].astype(np.float32),
                    "action": next_actions[frame].astype(np.float32),
                    "debug.block_pose": data["block_pose"][frame].astype(np.float32),
                    "debug.tcp_position": data["tcp_position"][frame].astype(np.float32),
                    "debug.phase": np.asarray([data["phase"][frame]], dtype=np.int64),
                    "next.done": np.asarray([last], dtype=np.bool_),
                    "next.success": np.asarray([last], dtype=np.bool_),
                    "task": args.task,
                })
        dataset.save_episode(parallel_encoding=False)
        total_frames += frame_count
        print(f"변환 {order}/{len(episodes)}: episode={index}, frames={frame_count}")

    dataset.finalize()
    loaded = LeRobotDataset(repo_id=args.repo_id, root=args.output)
    if loaded.num_episodes != len(episodes) or loaded.num_frames != total_frames:
        raise RuntimeError("변환 후 episode/frame 수 재검증에 실패했습니다.")
    sample = loaded[0]
    if tuple(sample[CAMERA_KEY].shape) != (3, height, width):
        raise RuntimeError(f"RGB tensor shape 오류: {sample[CAMERA_KEY].shape}")
    if "task" not in sample or not sample["task"]:
        raise RuntimeError("SmolVLA 언어 task가 저장되지 않았습니다.")
    print(f"변환/재로드 완료: {args.output}\nrepo_id={args.repo_id}, "
          f"episodes={loaded.num_episodes}, frames={loaded.num_frames}, fps={loaded.fps}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)
