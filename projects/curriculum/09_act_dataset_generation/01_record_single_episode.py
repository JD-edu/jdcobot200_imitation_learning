#!/usr/bin/env python3
"""무작위 안전 PnP 한 episode를 RGB/state/action으로 기록합니다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset_tools import episode_path, load_manifest, record_episode, sample_plan, save_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data/demo")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--path-steps", type=int, default=201)
    parser.add_argument("--max-sample-attempts", type=int, default=100)
    parser.add_argument("--max-place-error-mm", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.episode_index < 0 or args.fps < 1 or args.width < 32 or args.height < 32:
        raise ValueError("episode-index는 0 이상, fps는 1 이상, 영상 크기는 32 이상이어야 합니다.")
    path = episode_path(args.output, args.episode_index)
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"이미 존재합니다: {path} (--overwrite로 교체)")
    manifest = load_manifest(args.output)
    if manifest["fps"] not in (None, args.fps) or manifest["image_size"] not in (
        None, [args.height, args.width]
    ):
        raise RuntimeError("기존 데이터셋의 fps 또는 영상 크기와 다릅니다.")

    rng = np.random.default_rng(args.random_seed)
    start, target, pick_yaw, place_yaw, phases, attempts = sample_plan(
        rng, tolerance_mm=args.tolerance_mm, path_steps=args.path_steps,
        max_attempts=args.max_sample_attempts,
    )
    result = record_episode(
        path, episode_index=args.episode_index, start=start, target=target,
        pick_yaw=pick_yaw, place_yaw=place_yaw, phases=phases, fps=args.fps,
        width=args.width, height=args.height,
        max_place_error_mm=args.max_place_error_mm,
    )
    result["seed"] = args.random_seed
    result["sampling_attempts"] = attempts
    manifest["fps"] = args.fps
    manifest["image_size"] = [args.height, args.width]
    manifest["episodes"] = [
        item for item in manifest["episodes"]
        if item["episode_index"] != args.episode_index
    ] + [result]
    manifest["episodes"].sort(key=lambda item: item["episode_index"])
    save_manifest(args.output, manifest)
    print(f"기록 완료: {path}")
    print(f"frames={result['frames']}, 위치 오차={result['position_error_mm']:.3f}mm, "
          f"yaw 오차={result['yaw_error_deg']:.2f}°")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)
