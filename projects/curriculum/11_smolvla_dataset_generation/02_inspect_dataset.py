#!/usr/bin/env python3
"""11 데이터셋의 파일 무결성과 다양성 분포를 검사합니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def preview(dataset: Path, episodes: list[dict], selected: int | None,
            fps: int) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("--preview에는 opencv-python이 필요합니다.") from exc
    chosen = episodes if selected is None else [
        item for item in episodes if int(item["episode_index"]) == selected
    ]
    if not chosen:
        raise ValueError(f"episode {selected}가 없습니다.")
    delay_ms = max(1, int(round(1000 / fps)))
    for item in chosen:
        episode_index = int(item["episode_index"])
        path = dataset / "episodes" / f"episode_{episode_index:06d}.npz"
        with np.load(path) as data:
            for frame_index, rgb in enumerate(data["images"]):
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                phase = int(data["phase"][frame_index])
                cv2.putText(
                    bgr,
                    f"episode={episode_index} frame={frame_index + 1}/{len(data['images'])} phase={phase}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )
                cv2.imshow("SmolVLA dataset inspector (Q/ESC: stop)", bgr)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    return
    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "data/train")
    parser.add_argument("--preview", action="store_true",
                        help="검사 후 저장 영상을 GUI로 재생")
    parser.add_argument("--episode", type=int,
                        help="GUI로 볼 episode 번호(생략하면 전체)")
    parser.add_argument("--preview-fps", type=int,
                        help="재생 FPS(생략하면 manifest FPS)")
    args = parser.parse_args()
    manifest_path = args.dataset / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise ValueError("episode가 없습니다.")

    required = {"images", "observation_state", "action", "timestamp", "phase",
                "tcp_position", "block_pose"}
    total_frames = 0
    for item in episodes:
        path = args.dataset / "episodes" / f"episode_{int(item['episode_index']):06d}.npz"
        with np.load(path) as data:
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path.name}: 누락 필드 {sorted(missing)}")
            lengths = {key: len(data[key]) for key in required}
            if len(set(lengths.values())) != 1 or next(iter(lengths.values())) == 0:
                raise ValueError(f"{path.name}: frame 수 불일치 {lengths}")
            if data["observation_state"].shape[1:] != (6,) or data["action"].shape[1:] != (6,):
                raise ValueError(f"{path.name}: state/action shape 오류")
            total_frames += len(data["images"])

    def counts(key: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for item in episodes:
            value = str(item.get(key, "missing"))
            values[value] = values.get(value, 0) + 1
        return values

    print(json.dumps({
        "episodes": len(episodes), "frames": total_frames,
        "fps": manifest["fps"], "image_size": manifest["image_size"],
        "successes": sum(bool(item.get("success")) for item in episodes),
        "generation_attempts": manifest.get("total_generation_attempts"),
        "failed_attempts": len(manifest.get("failed_attempts", [])),
        "profiles": counts("interpolation_profile"),
        "speed_scales": counts("speed_scale"),
        "pick_cells": counts("pick_position_cell"),
        "place_cells": counts("place_position_cell"),
        "pick_yaw_bins": counts("pick_yaw_bin"),
        "place_yaw_bins": counts("place_yaw_bin"),
    }, ensure_ascii=False, indent=2))
    if args.episode is not None and not args.preview:
        raise ValueError("--episode는 --preview와 함께 사용해야 합니다.")
    if args.preview_fps is not None and args.preview_fps < 1:
        raise ValueError("--preview-fps는 1 이상이어야 합니다.")
    if args.preview:
        preview(args.dataset, episodes, args.episode,
                args.preview_fps or int(manifest["fps"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
