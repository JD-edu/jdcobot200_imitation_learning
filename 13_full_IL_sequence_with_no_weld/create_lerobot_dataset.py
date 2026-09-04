#!/usr/bin/env python3
"""Generate validated IK demonstrations and save one ready-to-train LeRobot dataset."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from generate_synthetic_trajectories import DEFAULT_XML, ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=25)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "synthetic_dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "lerobot_dataset")
    parser.add_argument("--repo-id", default="local/jdcobot200_contact_pick_place")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--min-floor-clearance", type=float, default=-0.008)
    parser.add_argument("--self-collision-margin", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.control_hz < 1 or args.horizon < 1:
        raise ValueError("episodes, control-hz, and horizon must be positive")
    if importlib.util.find_spec("lerobot") is None:
        raise ModuleNotFoundError(
            "LeRobot is required before generation starts. Install it with: "
            "python3 -m pip install 'lerobot>=0.4'"
        )

    generate = [
        sys.executable,
        str(ROOT / "generate_synthetic_trajectories.py"),
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--control-hz", str(args.control_hz),
        "--max-attempts", str(args.max_attempts),
        "--xml", str(args.xml),
        "--output-dir", str(args.work_dir),
        "--min-floor-clearance", str(args.min_floor_clearance),
        "--self-collision-margin", str(args.self_collision_margin),
    ]
    convert = [
        sys.executable,
        str(ROOT / "convert_to_lerobot.py"),
        "--input-dir", str(args.work_dir),
        "--output-dir", str(args.output_dir),
        "--repo-id", args.repo_id,
        "--xml", str(args.xml),
        "--horizon", str(args.horizon),
        "--frame-stride", str(args.frame_stride),
        "--width", str(args.width),
        "--height", str(args.height),
        "--video-backend", args.video_backend,
        "--image-writer-threads", str(args.image_writer_threads),
    ]
    if args.overwrite:
        generate.append("--overwrite")
        convert.append("--overwrite")

    run(generate)
    run(convert)
    print(f"Ready-to-train LeRobot dataset: {args.output_dir.resolve()}")
    print(
        f"Intermediate validated NPZ files kept at {args.work_dir.resolve()} "
        "for reproducibility."
    )


if __name__ == "__main__":
    main()
