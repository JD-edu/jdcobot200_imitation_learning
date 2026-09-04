#!/usr/bin/env python3
"""Validate a local LeRobot v3 dataset, including decoded training samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Hugging Face datasets creates lock files even when reading a local dataset.
# Use a writable cache so validation also works in restricted environments.
os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "lerobot_validation_hf"))


EXPECTED_JOINTS = [
    "base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"
]


class Report:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.failed: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        (self.passed if condition else self.failed).append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)

    def print(self) -> None:
        for message in self.passed:
            print(f"[PASS] {message}")
        for message in self.warnings:
            print(f"[WARN] {message}")
        for message in self.failed:
            print(f"[FAIL] {message}")
        print(
            f"\nResult: {len(self.passed)} passed, "
            f"{len(self.warnings)} warnings, {len(self.failed)} failed"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("lerobot_dataset"))
    parser.add_argument("--repo-id", default="local/jdcobot200_dual_camera_pick_place")
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--expected-fps", type=int, default=10)
    parser.add_argument("--expected-width", type=int, default=320)
    parser.add_argument("--expected-height", type=int, default=240)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of evenly spaced samples to decode and inspect",
    )
    return parser.parse_args()


def parquet_is_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as stream:
        header = stream.read(4)
        stream.seek(-4, os.SEEK_END)
        footer = stream.read(4)
    return header == b"PAR1" and footer == b"PAR1"


def feature_ok(features: dict[str, Any], key: str, dtype: str, shape: list[int]) -> bool:
    feature = features.get(key, {})
    return feature.get("dtype") == dtype and feature.get("shape") == shape


def validate_files(args: argparse.Namespace, report: Report) -> dict[str, Any] | None:
    root = args.root.resolve()
    report.check(root.is_dir(), f"dataset root exists: {root}")
    info_path = root / "meta" / "info.json"
    report.check(info_path.is_file(), "meta/info.json exists")
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        report.passed.append("meta/info.json is valid JSON")
    except Exception as error:
        report.failed.append(f"meta/info.json cannot be parsed: {error}")
        return None

    report.check(info.get("codebase_version") == "v3.0", "LeRobot codebase_version is v3.0")
    report.check(info.get("fps") == args.expected_fps, f"fps is {args.expected_fps}")
    report.check(
        info.get("total_episodes") == args.expected_episodes,
        f"total_episodes is {args.expected_episodes} (found {info.get('total_episodes')})",
    )
    report.check(int(info.get("total_frames", 0)) > 0, "total_frames is positive")

    features = info.get("features", {})
    report.check(feature_ok(features, "observation.state", "float32", [6]), "state feature is float32[6]")
    report.check(feature_ok(features, "action", "float32", [6]), "action feature is float32[6]")
    expected_image_shape = [3, args.expected_height, args.expected_width]
    for camera_name in ("front", "wrist"):
        image = features.get(f"observation.images.{camera_name}", {})
        report.check(image.get("dtype") == "video", f"{camera_name} camera uses video storage")
        report.check(image.get("shape") == expected_image_shape, f"{camera_name} camera shape is CHW {expected_image_shape}")
    report.check(
        features.get("observation.state", {}).get("names") == EXPECTED_JOINTS
        and features.get("action", {}).get("names") == EXPECTED_JOINTS,
        "state/action joint names and order are correct",
    )

    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    video_files = sorted((root / "videos").glob("**/*.mp4"))
    report.check(bool(parquet_files), "at least one data parquet exists")
    report.check(
        bool(parquet_files) and all(parquet_is_complete(path) for path in parquet_files),
        "all data parquet files have valid PAR1 header/footer",
    )
    report.check(bool(video_files), "at least one encoded camera video exists")
    report.check(
        bool(video_files) and all(path.stat().st_size > 0 for path in video_files),
        "all encoded camera videos are non-empty",
    )
    staging_images = list((root / "images").glob("**/*.png"))
    report.warn(not staging_images, f"temporary unencoded PNG frames remain ({len(staging_images)})")
    return info


def validate_loader(args: argparse.Namespace, report: Report, info: dict[str, Any]) -> None:
    expected_image_shape = [3, args.expected_height, args.expected_width]
    try:
        import numpy as np
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        offsets = [step / args.expected_fps for step in range(args.horizon)]
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.root.resolve(),
            delta_timestamps={"action": offsets},
            video_backend=args.video_backend,
        )
        report.passed.append("LeRobotDataset public loader opens the dataset")
    except Exception as error:
        report.failed.append(f"LeRobotDataset loader failed: {type(error).__name__}: {error}")
        return

    report.check(len(dataset) == info["total_frames"], "loader length equals metadata total_frames")
    sample_count = min(max(args.samples, 1), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=int).tolist()
    decoded = 0
    finite = True
    shapes = True
    dtypes = True
    for index in indices:
        try:
            sample = dataset[index]
            state = sample["observation.state"]
            action = sample["action"]
            cameras = [sample["observation.images.front"], sample["observation.images.wrist"]]
            pad = sample["action_is_pad"]
            shapes = shapes and tuple(state.shape) == (6,)
            shapes = shapes and tuple(action.shape) == (args.horizon, 6)
            shapes = shapes and all(tuple(camera.shape) == tuple(expected_image_shape) for camera in cameras)
            shapes = shapes and tuple(pad.shape) == (args.horizon,)
            dtypes = dtypes and state.dtype == torch.float32
            dtypes = dtypes and action.dtype == torch.float32
            finite = finite and bool(torch.isfinite(state).all() and torch.isfinite(action).all())
            finite = finite and all(bool(torch.isfinite(camera).all()) for camera in cameras)
            decoded += 1
        except Exception as error:
            report.failed.append(f"sample {index} decode failed: {type(error).__name__}: {error}")
            break
    report.check(decoded == sample_count, f"decoded {sample_count} distributed samples")
    report.check(shapes, "decoded state/action/image/padding shapes are correct")
    report.check(dtypes, "decoded state and action dtypes are float32")
    report.check(finite, "decoded state/action/image values are finite")


def main() -> None:
    args = parse_args()
    report = Report()
    info = validate_files(args, report)
    if info is not None:
        validate_loader(args, report, info)
    report.print()
    if report.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
