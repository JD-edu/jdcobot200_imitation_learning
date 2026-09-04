#!/usr/bin/env python3
"""MINK-based FK module for the jdcobot200 MuJoCo model.

This module is the FK counterpart of ``jdcobot200_mink_solver.py`` so that
imitation-learning policies see a symmetric IK/FK interface.

Pattern alignment with ``jdcobot200_mink_solver.SO101MinkSolver``:

    * ``mink.Configuration`` is the single source of state.
    * Frozen ``@dataclass`` reserves a stable contract for IL pipelines.
    * ``ARM_JOINT_NAMES`` lists the five arm joints in body order.
    * ``arm_qpos_addresses`` and ``arm_dof_addresses`` are cached in
      ``__init__`` so each ``solve_step`` call stays branch-free.
    * End-effector is resolved by name (default ``"graspframe"``).
    * ``set_target_position`` is optional but, when supplied, every
      ``FKResult`` carries a ``position_error`` - useful when validating an
      IK round trip.

The IK/FK pair share the same ``mink.Configuration`` update convention.
This means an IL policy outputs an EE target, IK returns joint angles,
and FK re-projects those joint angles back into the EE frame for
certification. Round-trip fidelity is the core invariant the policy
depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import mink
import mujoco
import numpy as np


@dataclass(frozen=True)
class FKResult:
    """Result of one FK evaluation on the five arm joints.

    Mirrors ``IKResult`` for the variables an IL pipeline cares about:
    * ``arm_qpos`` - the five arm joint angles used for the FK evaluation
    * ``position`` - end-effector position in world frame (metres)
    * ``orientation`` - end-effector 3x3 rotation matrix in world frame
    * ``quaternion`` - end-effector quaternion (w, x, y, z) in world frame
    * ``target_position`` - target the FK was checked against, if any
    * ``position_error`` - L2 distance from target, if a target was supplied
    """

    arm_qpos: np.ndarray
    position: np.ndarray
    orientation: np.ndarray
    quaternion: np.ndarray
    target_position: Optional[np.ndarray]
    position_error: Optional[float]


class SO101MinkFKSolver:
    """Forward kinematics solver restricted to the five SO-101 arm joints.

    Conventionally this sits next to ``SO101MinkSolver``:

        policy -> EE target -> IK -> joint angles -> FK -> EE pose

    The FK half validates that the IK half actually reached the EE pose
    requested by the policy, which is the audit trail every IL training
    run needs.
    """

    ARM_JOINT_NAMES = (
        "base",
        "shoulder",
        "elbow",
        "wrist_pitch",
        "wrist_roll",
    )

    def __init__(
        self,
        model: mujoco.MjModel,
        end_effector_site: str = "graspframe",
    ) -> None:
        self.model = model
        self.end_effector_site = end_effector_site
        self.configuration = mink.Configuration(model)

        self._site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE, end_effector_site
        )

        self.arm_joint_ids = np.array(
            [
                self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.ARM_JOINT_NAMES
            ],
            dtype=int,
        )
        self.arm_qpos_addresses = model.jnt_qposadr[self.arm_joint_ids].astype(int)
        self.arm_dof_addresses = model.jnt_dofadr[self.arm_joint_ids].astype(int)

        # Optional: populated via set_target_position() so a position_error
        # is included in every FKResult.
        self._target_position: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id == -1:
            raise ValueError(f"MuJoCo model does not contain '{name}'.")
        return int(object_id)

    # ------------------------------------------------------------------ #
    # State bootstrap
    # ------------------------------------------------------------------ #
    def reset(self, qpos: Iterable[float]) -> None:
        """Bootstrap the FK configuration from the current simulation qpos.

        Comparable to ``SO101MinkSolver.reset`` - both classes re-use the
        same ``mink.Configuration`` instance so subsequent ``solve_step``
        calls start from a warm configuration.
        """
        qpos_array = np.asarray(qpos, dtype=float)
        if qpos_array.shape != (self.model.nq,):
            raise ValueError(
                f"qpos shape must be ({self.model.nq},), "
                f"got {qpos_array.shape}."
            )
        self.configuration.update(qpos_array)

    def set_target_position(self, xyz: Optional[Iterable[float]]) -> None:
        """Optional. When set, every subsequent ``FKResult`` reports an error.

        Pass ``None`` to clear the target.
        """
        if xyz is None:
            self._target_position = None
            return
        xyz_array = np.asarray(xyz, dtype=float)
        if xyz_array.shape != (3,):
            raise ValueError("Target position must contain exactly x, y, z.")
        if not np.all(np.isfinite(xyz_array)):
            raise ValueError("Target position must contain finite values.")
        self._target_position = xyz_array.copy()

    # ------------------------------------------------------------------ #
    # Pose queries (independent of target setting)
    # ------------------------------------------------------------------ #
    def get_end_effector_position(self) -> np.ndarray:
        return self.configuration.data.site_xpos[self._site_id].copy()

    def get_end_effector_orientation(self) -> np.ndarray:
        """3x3 rotation matrix in world frame."""
        return self.configuration.data.site_xmat[self._site_id].reshape(
            3, 3
        ).copy()

    def get_end_effector_quaternion(self) -> np.ndarray:
        """4-vector quaternion (w, x, y, z) in world frame."""
        # MjData exposes a site's world orientation as ``site_xmat`` but
        # (unlike bodies) has no ``site_quat`` field. Convert that matrix
        # with MuJoCo's own convention to obtain a wxyz quaternion.
        quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(
            quaternion,
            self.configuration.data.site_xmat[self._site_id],
        )
        return quaternion

    # ------------------------------------------------------------------ #
    # One-shot FK evaluation
    # ------------------------------------------------------------------ #
    def solve_step(self, current_qpos: Iterable[float]) -> FKResult:
        """Run FK from the given qpos and return the EE pose.

        Unlike ``SO101MinkSolver.solve_step`` this is a one-shot
        evaluation: it does not integrate any velocity forward in time
        and does not touch ``mj_step``. It is therefore safe to call
        inside a forward-only data-collection loop without perturbing the
        MuJoCo integrator state.
        """
        current_qpos_array = np.asarray(current_qpos, dtype=float)
        if current_qpos_array.shape != (self.model.nq,):
            raise ValueError(
                f"qpos shape must be ({self.model.nq},), "
                f"got {current_qpos_array.shape}."
            )

        self.configuration.update(current_qpos_array)

        position = self.get_end_effector_position()
        orientation = self.get_end_effector_orientation()
        quaternion = self.get_end_effector_quaternion()

        arm_qpos = self.configuration.q[self.arm_qpos_addresses].copy()

        position_error: Optional[float] = None
        target_snapshot: Optional[np.ndarray] = None
        if self._target_position is not None:
            position_error = float(
                np.linalg.norm(self._target_position - position)
            )
            target_snapshot = self._target_position.copy()

        return FKResult(
            arm_qpos=arm_qpos,
            position=position,
            orientation=orientation,
            quaternion=quaternion,
            target_position=target_snapshot,
            position_error=position_error,
        )
