#!/usr/bin/env python3
"""여러 개의 무작위 안전 PnP episode를 합성 데이터셋으로 생성합니다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset_tools import episode_path, load_manifest, record_episode, sample_plan, save_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data/train")
    parser.add_argument("--episodes", type=int, default=100)
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
    if not 1 <= args.episodes <= 100000:
        raise ValueError("episodes는 1~100000 범위여야 합니다.")

    manifest = load_manifest(args.output)
    if manifest["fps"] not in (None, args.fps) or manifest["image_size"] not in (
        None, [args.height, args.width]
    ):
        raise RuntimeError("기존 데이터셋의 fps 또는 영상 크기와 다릅니다.")
    manifest["fps"] = args.fps
    manifest["image_size"] = [args.height, args.width]
    rng = np.random.default_rng(args.random_seed)
    entries = {item["episode_index"]: item for item in manifest["episodes"]}

    for index in range(args.episodes):
        path = episode_path(args.output, index)
        start, target, pick_yaw, place_yaw, phases, attempts = sample_plan(
            rng, tolerance_mm=args.tolerance_mm, path_steps=args.path_steps,
            max_attempts=args.max_sample_attempts,
        )
        if path.exists() and not args.overwrite:
            print(f"건너뜀 {index + 1}/{args.episodes}: {path.name}")
            continue
        result = record_episode(
            path, episode_index=index, start=start, target=target,
            pick_yaw=pick_yaw, place_yaw=place_yaw, phases=phases, fps=args.fps,
            width=args.width, height=args.height,
            max_place_error_mm=args.max_place_error_mm,
        )
        result["seed"] = args.random_seed
        result["sampling_attempts"] = attempts
        entries[index] = result
        manifest["episodes"] = [entries[key] for key in sorted(entries)]
        save_manifest(args.output, manifest)
        print(f"완료 {index + 1}/{args.episodes}: frames={result['frames']}, "
              f"error={result['position_error_mm']:.3f}mm/{result['yaw_error_deg']:.2f}°")
    print(f"데이터셋 생성 완료: {args.output} ({len(entries)} episodes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)
