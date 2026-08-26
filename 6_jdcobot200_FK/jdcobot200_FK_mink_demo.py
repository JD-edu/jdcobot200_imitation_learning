#!/usr/bin/env python3
"""Demonstration of ``SO101MinkFKSolver`` driven by fixed joint configurations.

A small set of fixed arm-joint poses is fed into the FK solver. For each
pose we evaluate the end-effector position and (when a target is supplied)
the L2 error against that target. This pattern matches the FK half of
an imitation-learning pipeline that works entirely in joint space.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

from jdcobot200_mink_fk_solver import SO101MinkFKSolver


XML_PATH = Path(__file__).resolve().parent / "jdcobot200.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    fk = SO101MinkFKSolver(model=model, end_effector_site="graspframe")

    # Bootstrap the FK configuration from the simulation qpos.
    home_qpos = data.qpos.copy()
    fk.reset(home_qpos)

    # Fixed joint poses (rad) used as the FK sweep. Each row is the five
    # arm joint angles in (base, shoulder, elbow, wrist_pitch, wrist_roll)
    # order - the order declared in SO101MinkFKSolver.ARM_JOINT_NAMES.
    arm_pose_sequence = np.array(
        [
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.30, -0.20, 0.40, -0.30, 0.00],
            [-0.45, 0.15, -0.30, 0.40, 0.10],
            [0.10, -0.55, 0.55, -0.20, -0.30],
        ]
    )

    # Optional verification targets so the FK prints a position_error
    # against a known reachable EE pose. These are deliberately approximate
    # to keep the FK sweep focused on the FK side itself.
    target_sequence = np.array(
        [
            [0.2736, 0.0000, 0.3077],
            [0.3024, 0.1494, 0.1424],
            [0.1767, -0.2056, 0.2219],
            [0.1823, 0.0143, 0.1172],
        ]
    )

    print("FK sweep: fixed arm pose -> EE position"            )
    print(
        f"{'iter':>4} {'arm qpos (rad)':>36} "
        f"{'FK xyz (m)':>22} {'err mm':>10}"
    )
    print("-" * 80)

    last_fk_result = None
    for k, (arm_qpos, target) in enumerate(zip(arm_pose_sequence, target_sequence)):
        qpos = data.qpos.copy()
        qpos[fk.arm_qpos_addresses] = arm_qpos

        fk.set_target_position(target)
        fk_result = fk.solve_step(current_qpos=qpos)

        assert fk_result.position_error is not None
        print(
            f"{k:>4d} "
            f"{np.round(arm_qpos, 3)!s:>36} "
            f"{np.round(fk_result.position, 3)!s:>22} "
            f"{fk_result.position_error * 1000.0:>10.2f}"
        )
        last_fk_result = fk_result

    # Final pose detail exposes the FK side's full output (rotation plus
    # quaternion plus arm_qpos) in a single block.
    assert last_fk_result is not None
    fk_final = last_fk_result

    print()
    print("Final FK pose detail")
    print(f"  arm_qpos    (rad): {np.round(fk_final.arm_qpos, 4)}")
    print(f"  EE position (m)  : {np.round(fk_final.position, 5)}")
    print(f"  EE quat (wxyz)   : {np.round(fk_final.quaternion, 4)}")
    print(
        "  EE rotation (3x3):\n"
        f"{np.round(fk_final.orientation, 3)}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
