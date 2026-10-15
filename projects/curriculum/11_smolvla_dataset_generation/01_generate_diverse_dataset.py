#!/usr/bin/env python3
"""SmolVLA용으로 다양성을 높인 MuJoCo/IK pick-and-place 데이터셋을 생성합니다.

09의 검증된 IK, MuJoCo 접촉 물리, NPZ schema를 그대로 사용합니다. 각 episode는
pick/place 위치 strata, 두 yaw strata, 속도 및 joint 보간 profile을 순환하므로 단순
무작위 표본보다 작업공간과 동작 형태를 고르게 덮습니다.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from typing import Callable, Iterator

import numpy as np


LESSON_DIR = Path(__file__).resolve().parent
CURRICULUM_DIR = LESSON_DIR.parent
TOOLS_PATH = CURRICULUM_DIR / "09_act_dataset_generation/dataset_tools.py"


def load_dataset_tools():
    spec = importlib.util.spec_from_file_location("curriculum09_dataset_tools", TOOLS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"09 데이터 도구를 읽을 수 없습니다: {TOOLS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOLS = load_dataset_tools()
EXPERT = TOOLS.EXPERT


PROFILES: dict[str, Callable[[float], float]] = {
    "smoothstep": EXPERT.smoothstep,
}


def preview_episode(path: Path, fps: int, episode_index: int) -> bool:
    """저장된 dataset_camera 영상을 GUI로 재생합니다. False면 사용자가 종료했습니다."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("--preview에는 opencv-python이 필요합니다.") from exc

    delay_ms = max(1, int(round(1000 / fps)))
    with np.load(path) as data:
        for frame_index, rgb in enumerate(data["images"]):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr, f"episode={episode_index} frame={frame_index + 1}/{len(data['images'])}",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                cv2.LINE_AA,
            )
            cv2.imshow("SmolVLA dataset preview (Q/ESC: stop)", bgr)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return False
    cv2.destroyAllWindows()
    return True


def stratum(bounds: tuple[float, float], index: int, count: int) -> tuple[float, float]:
    edges = np.linspace(bounds[0], bounds[1], count + 1)
    return float(edges[index]), float(edges[index + 1])


@contextmanager
def episode_variation(index: int, grid_size: int, yaw_bins: int,
                      speed_scale: float, profile_name: str) -> Iterator[None]:
    """09 expert의 sampling 범위와 시간 profile을 한 episode 동안만 변경합니다."""
    original = {
        "pick_x": EXPERT.PICK_X_RANGE,
        "pick_y": EXPERT.PICK_Y_RANGE,
        "place_x": EXPERT.PLACE_X_RANGE,
        "place_y": EXPERT.PLACE_Y_RANGE,
        "smoothstep": EXPERT.smoothstep,
        "plan_episode": EXPERT.plan_episode,
    }
    pick_cell = index % (grid_size * grid_size)
    place_cell = (index * 7 + 3) % (grid_size * grid_size)
    EXPERT.PICK_X_RANGE = stratum(original["pick_x"], pick_cell % grid_size, grid_size)
    EXPERT.PICK_Y_RANGE = stratum(original["pick_y"], pick_cell // grid_size, grid_size)
    EXPERT.PLACE_X_RANGE = stratum(original["place_x"], place_cell % grid_size, grid_size)
    EXPERT.PLACE_Y_RANGE = stratum(original["place_y"], place_cell // grid_size, grid_size)
    EXPERT.smoothstep = PROFILES[profile_name]

    # 원래 IK 경로를 사용하면서 phase별 시간을 같은 비율로 변화시킨다.
    def varied_plan(*args, **kwargs):
        phases = original["plan_episode"](*args, **kwargs)
        return [
            EXPERT.Phase(p.name, p.start_q, p.goal_q, p.duration * speed_scale)
            for p in phases
        ]

    EXPERT.plan_episode = varied_plan

    # sample_valid_episode의 yaw도 균일 난수 대신 순환 stratum 안에서 뽑도록 RNG
    # 자체는 그대로 두며, 아래 range 교체는 위치에만 적용된다. yaw stratum은 생성 후
    # 실패 시 재시도까지 안전하게 처리하기 위해 sample wrapper에서 사용한다.
    try:
        yield
    finally:
        EXPERT.PICK_X_RANGE = original["pick_x"]
        EXPERT.PICK_Y_RANGE = original["pick_y"]
        EXPERT.PLACE_X_RANGE = original["place_x"]
        EXPERT.PLACE_Y_RANGE = original["place_y"]
        EXPERT.smoothstep = original["smoothstep"]
        EXPERT.plan_episode = original["plan_episode"]


def sample_stratified_plan(rng: np.random.Generator, index: int, yaw_bins: int,
                           tolerance_mm: float, path_steps: int,
                           max_attempts: int):
    """두 yaw를 지정 stratum에서 뽑고 09와 동일한 IK/FK/path 검사를 수행합니다."""
    solver = EXPERT.SafeIK()
    elbow = EXPERT.JOINT_NAMES.index("elbow")
    solver.fk.lower[elbow] = np.deg2rad(EXPERT.ELBOW_LOWER_DEG)
    pick_bin = index % yaw_bins
    place_bin = (index * 5 + 1) % yaw_bins
    pick_range = stratum((-np.pi / 2, np.pi / 2), pick_bin, yaw_bins)
    place_range = stratum((-np.pi / 2, np.pi / 2), place_bin, yaw_bins)
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        start = np.array([rng.uniform(*EXPERT.PICK_X_RANGE),
                          rng.uniform(*EXPERT.PICK_Y_RANGE), EXPERT.BLOCK_Z])
        target = np.array([rng.uniform(*EXPERT.PLACE_X_RANGE),
                           rng.uniform(*EXPERT.PLACE_Y_RANGE), EXPERT.BLOCK_Z])
        pick_yaw = rng.uniform(*pick_range)
        place_yaw = rng.uniform(*place_range)
        try:
            waypoints = EXPERT.make_waypoints(start, target, pick_yaw, place_yaw)
            cosine, sine = np.cos(place_yaw), np.sin(place_yaw)
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            offset = rotation @ TOOLS.PHYSICS_PLACE_POSITION_COMPENSATION_LOCAL_M
            for name in ("place_approach", "place", "retreat"):
                waypoints[name][:2] -= offset
            phases = EXPERT.plan_episode(
                solver, waypoints, pick_yaw,
                place_yaw - TOOLS.PHYSICS_PLACE_YAW_COMPENSATION_RAD,
                tolerance_mm=tolerance_mm, path_steps=path_steps,
            )
            return start, target, pick_yaw, place_yaw, phases, attempt
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            last_error = str(exc)
    raise RuntimeError(f"episode {index}: {max_attempts}회 IK 재시도 실패: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LESSON_DIR / "data/train")
    parser.add_argument("--episodes", type=int, default=50,
                        help="SmolVLA 공식 권장 시작 수량인 50 episode")
    parser.add_argument("--random-seed", type=int, default=1100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--yaw-bins", type=int, default=8)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--path-steps", type=int, default=201)
    parser.add_argument("--max-sample-attempts", type=int, default=100)
    parser.add_argument("--max-total-attempts", type=int,
                        help="성공 episode를 채우기 위한 전체 시도 한도(기본: episodes의 10배)")
    parser.add_argument("--max-place-error-mm", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview", action="store_true",
                        help="각 episode 생성 직후 dataset camera 영상을 GUI로 재생")
    args = parser.parse_args()
    if args.episodes < 1 or args.grid_size < 1 or args.yaw_bins < 1:
        raise ValueError("episodes, grid-size, yaw-bins는 1 이상이어야 합니다.")
    max_total_attempts = args.max_total_attempts or args.episodes * 10
    if max_total_attempts < args.episodes:
        raise ValueError("max-total-attempts는 episodes 이상이어야 합니다.")

    manifest = TOOLS.load_manifest(args.output)
    if manifest["fps"] not in (None, args.fps) or manifest["image_size"] not in (
        None, [args.height, args.width]
    ):
        raise RuntimeError("기존 데이터셋과 fps 또는 영상 크기가 다릅니다.")
    manifest["fps"] = args.fps
    manifest["image_size"] = [args.height, args.width]
    manifest["generation_mode"] = "mujoco_contact_physics_stratified_ik"
    manifest["diversity"] = {
        "position_grid": [args.grid_size, args.grid_size],
        "yaw_bins": args.yaw_bins,
        # 접촉 물리 보정값은 09에서 검증한 이 timing/profile을 전제로 한다.
        # 다양성은 위치와 yaw strata에서 확보하고 timing은 임의 변경하지 않는다.
        "speed_scales": [1.0],
        "interpolation_profiles": list(PROFILES),
    }
    entries = {int(item["episode_index"]): item for item in manifest["episodes"]}
    failed_attempts = manifest.setdefault("failed_attempts", [])
    total_attempts = int(manifest.get("total_generation_attempts", 0))
    if args.overwrite:
        # 같은 base seed에서 완전히 재현 가능한 재생성을 시작한다.
        failed_attempts.clear()
        total_attempts = 0
    speeds = (1.0,)
    profile_names = tuple(PROFILES)

    index = 0
    stopped_by_user = False
    while index < args.episodes:
        path = TOOLS.episode_path(args.output, index)
        if path.exists() and not args.overwrite:
            print(f"건너뜀 {index + 1}/{args.episodes}: {path.name}")
            index += 1
            continue
        if total_attempts >= max_total_attempts:
            raise RuntimeError(
                f"전체 시도 한도 도달: successes={len(entries)}/{args.episodes}, "
                f"attempts={total_attempts}/{max_total_attempts}. "
                "--max-total-attempts를 늘리거나 manifest의 실패 원인을 확인하세요."
            )

        attempt_seed = args.random_seed + total_attempts
        rng = np.random.default_rng(attempt_seed)
        total_attempts += 1
        # 실패 시 seed만 바꾸고 같은 어려운 cell/yaw 조합에 갇히지 않도록
        # sampling variation도 전체 시도 번호에 따라 함께 순환한다.
        variation_index = total_attempts - 1
        speed = speeds[variation_index % len(speeds)]
        profile = profile_names[(variation_index // len(speeds)) % len(profile_names)]
        try:
            with episode_variation(
                variation_index, args.grid_size, args.yaw_bins, speed, profile
            ):
                start, target, pick_yaw, place_yaw, phases, attempts = sample_stratified_plan(
                    rng, variation_index, args.yaw_bins, args.tolerance_mm, args.path_steps,
                    args.max_sample_attempts,
                )
                result = TOOLS.record_episode(
                    path, episode_index=index, start=start, target=target,
                    pick_yaw=pick_yaw, place_yaw=place_yaw, phases=phases,
                    fps=args.fps, width=args.width, height=args.height,
                    max_place_error_mm=args.max_place_error_mm,
                )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            failure = {
                "attempt": total_attempts,
                "episode_index": index,
                "seed": attempt_seed,
                "reason": str(exc),
            }
            failed_attempts.append(failure)
            manifest["failed_attempts"] = failed_attempts
            manifest["total_generation_attempts"] = total_attempts
            manifest["episodes"] = [entries[key] for key in sorted(entries)]
            TOOLS.save_manifest(args.output, manifest)
            print(
                f"실패 후 재시도: target={index + 1}/{args.episodes}, "
                f"attempt={total_attempts}/{max_total_attempts}, "
                f"seed={attempt_seed}, reason={exc}"
            )
            continue
        result.update(seed=attempt_seed, base_seed=args.random_seed,
                      sampling_attempts=attempts,
                      attempt_seed=attempt_seed, generation_attempt=total_attempts,
                      variation_index=variation_index,
                      speed_scale=speed, interpolation_profile=profile,
                      pick_position_cell=variation_index % (args.grid_size**2),
                      place_position_cell=(variation_index * 7 + 3) % (args.grid_size**2),
                      pick_yaw_bin=variation_index % args.yaw_bins,
                      place_yaw_bin=(variation_index * 5 + 1) % args.yaw_bins)
        entries[index] = result
        manifest["episodes"] = [entries[key] for key in sorted(entries)]
        manifest["failed_attempts"] = failed_attempts
        manifest["total_generation_attempts"] = total_attempts
        TOOLS.save_manifest(args.output, manifest)
        print(f"성공 {index + 1}/{args.episodes}: frames={result['frames']}, "
              f"attempt={total_attempts}, seed={attempt_seed}, "
              f"profile={profile}, speed={speed:.2f}, "
              f"error={result['position_error_mm']:.2f}mm")
        if args.preview and not preview_episode(path, args.fps, index):
            print("사용자가 preview를 종료했습니다. 생성 작업도 안전하게 중단합니다.")
            stopped_by_user = True
            break
        index += 1

    if stopped_by_user:
        print(f"생성 중단: {args.output} (successes={len(entries)}/{args.episodes}, "
              f"failures={len(failed_attempts)}, attempts={total_attempts})")
    else:
        print(f"생성 완료: {args.output} (successes={len(entries)}/{args.episodes}, "
              f"failures={len(failed_attempts)}, attempts={total_attempts})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)
