#!/usr/bin/env python3
"""Map the JDCobot200 collision-safe workspace with MINK-based FK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from safety_model import JDCobotSafetyModel, ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MINK FK 기반 JDCobot200 safe workspace 분석"
    )
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--collision-margin-mm", type=float, default=5.0)
    parser.add_argument("--floor-margin-mm", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples는 양수여야 합니다.")
    if args.collision_margin_mm < 0 or args.floor_margin_mm < 0:
        raise ValueError("margin은 0 이상이어야 합니다.")

    safety = JDCobotSafetyModel()
    rng = np.random.default_rng(args.seed)
    joint_samples = rng.uniform(safety.lower, safety.upper, size=(args.samples, 5))
    positions = np.empty((args.samples, 3))
    safe_mask = np.zeros(args.samples, dtype=bool)
    self_clearance = np.empty(args.samples)
    floor_clearance = np.empty(args.samples)
    reason_counts: dict[str, int] = {}
    collision_margin = args.collision_margin_mm / 1000.0
    floor_margin = args.floor_margin_mm / 1000.0

    for index, q in enumerate(joint_samples):
        result = safety.evaluate(q, collision_margin, floor_margin)
        positions[index] = result.tcp_position
        safe_mask[index] = result.safe
        self_clearance[index] = result.minimum_self_clearance
        floor_clearance[index] = result.minimum_floor_clearance
        for reason in result.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if (index + 1) % max(1, args.samples // 10) == 0:
            print(f"진행률 {index + 1:,}/{args.samples:,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "safe_zone_samples.npz",
        q=joint_samples,
        tcp_xyz=positions,
        safe=safe_mask,
        self_clearance_m=self_clearance,
        floor_clearance_m=floor_clearance,
    )
    summary = {
        "fk_backend": "mink.Configuration",
        "samples": args.samples,
        "seed": args.seed,
        "safe_count": int(np.sum(safe_mask)),
        "unsafe_count": int(np.sum(~safe_mask)),
        "safe_ratio": float(np.mean(safe_mask)),
        "collision_margin_mm": args.collision_margin_mm,
        "floor_margin_mm": args.floor_margin_mm,
        "unsafe_reasons": reason_counts,
        "safe_tcp_min_m": positions[safe_mask].min(axis=0).tolist()
        if np.any(safe_mask) else None,
        "safe_tcp_max_m": positions[safe_mask].max(axis=0).tolist()
        if np.any(safe_mask) else None,
    }
    (args.output_dir / "safe_zone_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure = plt.figure(figsize=(11, 9))
    axis = figure.add_subplot(111, projection="3d")
    unsafe = positions[~safe_mask]
    safe = positions[safe_mask]
    if len(unsafe):
        axis.scatter(*unsafe.T, c="red", s=1, alpha=0.08,
                     label=f"unsafe ({len(unsafe):,})")
    if len(safe):
        axis.scatter(*safe.T, c="green", s=2, alpha=0.35,
                     label=f"safe ({len(safe):,})")
    axis.set(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
        title="JDCobot200 MINK FK collision-safe workspace",
    )
    axis.legend()
    axis.set_box_aspect((1, 1, 1))
    figure.tight_layout()
    figure.savefig(args.output_dir / "safe_zone_workspace.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    print(f"결과 저장: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
