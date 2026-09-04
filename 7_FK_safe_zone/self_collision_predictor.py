#!/usr/bin/env python3
"""Predict self/floor collision clearance for a pose or joint trajectory."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from safety_model import ARM_JOINT_NAMES, JDCobotSafetyModel


def five_degrees(values: list[float] | None, option: str) -> np.ndarray:
    if values is None or len(values) != 5:
        raise ValueError(f"{option}에는 관절각 5개가 필요합니다.")
    return np.deg2rad(np.asarray(values, dtype=float))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JDCobot200 링크 충돌 예측")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--q-deg", nargs=5, type=float, metavar=("J1", "J2", "J3", "J4", "J5"))
    group.add_argument("--goal-deg", nargs=5, type=float, metavar=("J1", "J2", "J3", "J4", "J5"))
    parser.add_argument("--start-deg", nargs=5, type=float, default=[0, 0, 0, 0, 0])
    parser.add_argument("--steps", type=int, default=201)
    parser.add_argument("--collision-margin-mm", type=float, default=5.0)
    parser.add_argument("--floor-margin-mm", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 2 or args.collision_margin_mm < 0 or args.floor_margin_mm < 0:
        raise ValueError("steps는 2 이상이고 margin은 0 이상이어야 합니다.")
    safety = JDCobotSafetyModel()
    if args.q_deg is not None:
        trajectory = five_degrees(args.q_deg, "--q-deg")[None, :]
    else:
        start = five_degrees(args.start_deg, "--start-deg")
        goal = five_degrees(args.goal_deg, "--goal-deg")
        trajectory = np.linspace(start, goal, args.steps)

    worst_self = (float("inf"), 0, ("", ""))
    worst_floor = (float("inf"), 0, "")
    first_unsafe = None
    for index, q in enumerate(trajectory):
        result = safety.evaluate(
            q, args.collision_margin_mm / 1000.0, args.floor_margin_mm / 1000.0
        )
        if result.minimum_self_clearance < worst_self[0]:
            worst_self = (result.minimum_self_clearance, index, result.closest_self_pair)
        if result.minimum_floor_clearance < worst_floor[0]:
            worst_floor = (result.minimum_floor_clearance, index, result.closest_floor_geom)
        if not result.safe and first_unsafe is None:
            first_unsafe = (index, q.copy(), result.reasons)

    print("관절 순서:", ", ".join(ARM_JOINT_NAMES))
    print(f"검사 자세 수: {len(trajectory)}")
    print(f"최소 self clearance: {worst_self[0] * 1000:.2f} mm "
          f"(step {worst_self[1]}, {worst_self[2][0]} / {worst_self[2][1]})")
    print(f"최소 floor clearance: {worst_floor[0] * 1000:.2f} mm "
          f"(step {worst_floor[1]}, {worst_floor[2]})")
    if first_unsafe is not None:
        index, q, reasons = first_unsafe
        print(f"UNSAFE: 최초 위험 step={index}, q(deg)={np.round(np.rad2deg(q), 3)}, "
              f"reason={','.join(reasons)}")
        return 2
    print("SAFE: 지정한 자세/궤적은 설정 margin을 만족합니다.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"입력 오류: {exc}", file=sys.stderr)
        sys.exit(1)
