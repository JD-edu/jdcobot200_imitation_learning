#!/usr/bin/env python3
"""09 데이터셋의 배열, 시간 동기화 및 episode 영상을 검사합니다."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset_tools import episode_path, load_manifest


def validate_episode(path: Path, fps: int, image_size: list[int]) -> tuple[int, float]:
    with np.load(path) as episode:
        required = {"images", "observation_state", "action", "timestamp", "phase",
                    "tcp_position", "block_pose"}
        missing = required - set(episode.files)
        if missing:
            raise RuntimeError(f"{path.name}: 필드 누락 {sorted(missing)}")
        lengths = {name: len(episode[name]) for name in required}
        if len(set(lengths.values())) != 1 or not next(iter(lengths.values())):
            raise RuntimeError(f"{path.name}: frame 길이가 일치하지 않습니다: {lengths}")
        frames = lengths["images"]
        if episode["images"].shape[1:] != (*image_size, 3):
            raise RuntimeError(f"{path.name}: 영상 shape 오류 {episode['images'].shape}")
        if episode["observation_state"].shape != (frames, 6):
            raise RuntimeError(f"{path.name}: state shape 오류")
        if episode["action"].shape != (frames, 6):
            raise RuntimeError(f"{path.name}: action shape 오류")
        delta = np.diff(episode["timestamp"])
        if len(delta) and not np.allclose(delta, 1 / fps, atol=1e-8):
            raise RuntimeError(f"{path.name}: timestamp 간격 오류")
        return frames, float(episode["timestamp"][-1])


def replay(path: Path, fps: int, export: Path | None) -> None:
    with np.load(path) as episode:
        images = episode["images"]
        phases = episode["phase"]
        writer = None
        if export is not None:
            export.parent.mkdir(parents=True, exist_ok=True)
            height, width = images.shape[1:3]
            writer = cv2.VideoWriter(str(export), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"MP4를 열 수 없습니다: {export}")
        try:
            for index, rgb in enumerate(images):
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, f"frame={index} phase={int(phases[index])}",
                            (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                if writer is not None:
                    writer.write(bgr)
                else:
                    cv2.imshow("ACT dataset replay (ESC: stop)", bgr)
                    if cv2.waitKey(max(1, round(1000 / fps))) == 27:
                        break
        finally:
            if writer is not None:
                writer.release()
            else:
                cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "data/train")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--export-mp4", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.dataset)
    if not manifest["episodes"] or manifest["fps"] is None:
        raise RuntimeError(f"episode가 없는 데이터셋입니다: {args.dataset}")
    if manifest.get("generation_mode") == "mujoco_contact_physics":
        for item in manifest["episodes"]:
            if not item.get("physics_contact") or not item.get("bilateral_grasp_contact"):
                raise RuntimeError(
                    f"episode {item['episode_index']}: 물리 contact 성공 기록이 없습니다."
                )
            if item.get("max_block_height_m", 0.0) < 0.08:
                raise RuntimeError(
                    f"episode {item['episode_index']}: 블록 들어올림 높이가 부족합니다."
                )
    total_frames = 0
    for item in manifest["episodes"]:
        path = episode_path(args.dataset, item["episode_index"])
        if not path.exists():
            raise RuntimeError(f"manifest 파일이 없습니다: {path}")
        frames, _duration = validate_episode(path, manifest["fps"], manifest["image_size"])
        total_frames += frames
    print(f"검사 통과: episodes={len(manifest['episodes'])}, frames={total_frames}, "
          f"fps={manifest['fps']}, image={manifest['image_size']}")
    if args.replay or args.export_mp4:
        index = args.episode if args.episode is not None else manifest["episodes"][0]["episode_index"]
        path = episode_path(args.dataset, index)
        if not path.exists():
            raise RuntimeError(f"episode가 없습니다: {path}")
        replay(path, manifest["fps"], args.export_mp4)
        if args.export_mp4:
            print(f"영상 저장 완료: {args.export_mp4}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)
