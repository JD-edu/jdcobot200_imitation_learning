#!/usr/bin/env python3
"""09 NPZ 원본을 LeRobot v0.4.4 Dataset v3 형식으로 변환합니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
CAMERA_KEY = "observation.images.dataset_camera"
TASK = "Pick the red block and place it on the green target."


def read_manifest(source: Path) -> dict:
    path = source / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"09 manifest가 없습니다: {path}")
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not manifest.get("episodes"):
        raise ValueError("변환할 episode가 없습니다.")
    return manifest


def make_features(height: int, width: int) -> dict:
    return {
        CAMERA_KEY: {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        # 아래 항목은 이름이 observation/action으로 시작하지 않아 ACT 입력에서 제외된다.
        "debug.block_pose": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "z", "yaw"],
        },
        "debug.tcp_position": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "debug.phase": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["phase"],
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["done"],
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["success"],
        },
    }


def validate_npz(episode: np.lib.npyio.NpzFile, path: Path,
                 height: int, width: int) -> int:
    required = {
        "images", "observation_state", "action", "timestamp", "phase",
        "tcp_position", "block_pose",
    }
    missing = required - set(episode.files)
    if missing:
        raise ValueError(f"{path.name}: 필드 누락 {sorted(missing)}")
    frames = len(episode["images"])
    if frames == 0 or any(len(episode[key]) != frames for key in required):
        raise ValueError(f"{path.name}: 배열 frame 수가 일치하지 않습니다.")
    expected = {
        "images": (frames, height, width, 3),
        "observation_state": (frames, 6),
        "action": (frames, 6),
        "phase": (frames,),
        "tcp_position": (frames, 3),
        "block_pose": (frames, 4),
    }
    for key, shape in expected.items():
        if episode[key].shape != shape:
            raise ValueError(
                f"{path.name}: {key} shape={episode[key].shape}, expected={shape}"
            )
    return frames


def convert(source: Path, output: Path, repo_id: str) -> None:
    manifest = read_manifest(source)
    if output.exists():
        raise FileExistsError(
            f"출력 경로가 이미 존재합니다: {output}\n"
            "LeRobot 생성 규칙과 기존 데이터 보호를 위해 존재하지 않는 새 경로를 지정하세요."
        )
    fps = int(manifest["fps"])
    height, width = (int(value) for value in manifest["image_size"])
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output,
        robot_type="jdcobot200_mujoco",
        features=make_features(height, width),
        use_videos=False,
    )

    total_frames = 0
    for order, metadata in enumerate(manifest["episodes"]):
        index = int(metadata["episode_index"])
        path = source / "episodes" / f"episode_{index:06d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"episode 파일이 없습니다: {path}")
        with np.load(path) as episode:
            frames = validate_npz(episode, path, height, width)
            # 09는 command 적용 뒤 state/action을 기록했다. LeRobot record loop와 동일하게
            # observation(t)에 실행할 다음 명령을 대응시키기 위해 action을 한 frame 이동한다.
            next_actions = np.concatenate(
                [episode["action"][1:], episode["action"][-1:]], axis=0
            ).astype(np.float32)
            for frame_index in range(frames):
                last = frame_index == frames - 1
                dataset.add_frame({
                    CAMERA_KEY: episode["images"][frame_index],
                    "observation.state": episode["observation_state"][frame_index].astype(np.float32),
                    "action": next_actions[frame_index],
                    "debug.block_pose": episode["block_pose"][frame_index].astype(np.float32),
                    "debug.tcp_position": episode["tcp_position"][frame_index].astype(np.float32),
                    "debug.phase": np.asarray([episode["phase"][frame_index]], dtype=np.int64),
                    "next.done": np.asarray([last], dtype=np.bool_),
                    "next.success": np.asarray([last and bool(metadata["success"])], dtype=np.bool_),
                    "task": TASK,
                })
        dataset.save_episode(parallel_encoding=False)
        total_frames += frames
        print(f"변환 {order + 1}/{len(manifest['episodes'])}: episode={index}, frames={frames}")

    dataset.finalize()
    loaded = LeRobotDataset(repo_id=repo_id, root=output)
    if loaded.num_episodes != len(manifest["episodes"]) or loaded.num_frames != total_frames:
        raise RuntimeError(
            f"재검증 실패: episodes={loaded.num_episodes}, frames={loaded.num_frames}"
        )
    first = loaded[0]
    if tuple(first[CAMERA_KEY].shape) != (3, height, width):
        raise RuntimeError(f"LeRobot RGB load shape 오류: {first[CAMERA_KEY].shape}")
    print(
        f"LeRobot v3 변환 및 재로드 검증 완료: {output}\n"
        f"repo_id={repo_id}, episodes={loaded.num_episodes}, "
        f"frames={loaded.num_frames}, fps={loaded.fps}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="09 NPZ 데이터셋 폴더(manifest.json이 있는 위치)")
    parser.add_argument("--output", type=Path, required=True,
                        help="새 LeRobot 데이터셋을 만들 빈 폴더")
    parser.add_argument("--repo-id", default="local/jdcobot200_synthetic_pnp",
                        help="LeRobot dataset 식별자(namespace/name)")
    args = parser.parse_args()
    if "/" not in args.repo_id or args.repo_id.startswith("/") or args.repo_id.endswith("/"):
        raise ValueError("repo-id는 namespace/name 형식이어야 합니다.")
    convert(args.source.resolve(), args.output.resolve(), args.repo_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)
