#!/usr/bin/env python3
"""MINK solver module for the uploaded SO-101 MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mink
import mujoco
import numpy as np


@dataclass(frozen=True)
class IKResult:
    arm_qpos: np.ndarray
    position_error: float
    current_position: np.ndarray
    target_position: np.ndarray
    solution_found: bool


class SO101MinkSolver:
    """Differential IK solver restricted to the five SO-101 arm joints.

    The gripper joint and the blue block free joint are not allowed to move in
    the IK optimization result. The solver accepts the current MuJoCo qpos and
    returns desired qpos values for the five arm joints.
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
        solver: str = "daqp",
        position_cost: float = 1.0,
        orientation_cost: float = 0.0,
        posture_cost: float = 1e-3,
        lm_damping: float = 1e-3,
        max_joint_velocity: float = 2.0,
    ) -> None:
        self.model = model
        self.configuration = mink.Configuration(model)
        self.end_effector_site = end_effector_site
        self.solver = solver
        self.max_joint_velocity = float(max_joint_velocity)

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
        arm_actuator_ids = np.array(
            [
                self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.ARM_JOINT_NAMES
            ],
            dtype=int,
        )
        self.arm_ctrl_ranges = model.actuator_ctrlrange[arm_actuator_ids].copy()

        self.frame_task = mink.FrameTask(
            frame_name=end_effector_site,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping,
        )
        self.posture_task = mink.PostureTask(model=model, cost=posture_cost)
        self.limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(
                model=model,
                velocities={
                    name: self.max_joint_velocity for name in self.ARM_JOINT_NAMES
                },
            ),
        ]

        self._target_position = np.zeros(3, dtype=float)
        self._initialized = False

    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id == -1:
            raise ValueError(f"MuJoCo model does not contain '{name}'.")
        return int(object_id)

    def reset(self, qpos: Iterable[float]) -> None:
        """Reset the IK configuration from the current simulation qpos."""
        qpos_array = np.asarray(qpos, dtype=float)
        if qpos_array.shape != (self.model.nq,):
            raise ValueError(
                f"qpos shape must be ({self.model.nq},), got {qpos_array.shape}."
            )

        self.configuration.update(qpos_array)
        self.posture_task.set_target_from_configuration(self.configuration)
        self._initialized = True

    def set_target_position(self, xyz: Iterable[float]) -> None:
        xyz_array = np.asarray(xyz, dtype=float)
        if xyz_array.shape != (3,):
            raise ValueError("Target position must contain exactly x, y, z.")
        if not np.all(np.isfinite(xyz_array)):
            raise ValueError("Target position must contain finite values.")

        self._target_position = xyz_array.copy()
        self.frame_task.set_target(mink.SE3.from_translation(self._target_position))

    def get_end_effector_position(self) -> np.ndarray:
        return self.configuration.data.site_xpos[self._site_id].copy()

    def solve_step(
        self,
        current_qpos: Iterable[float],
        dt: float,
        iterations: int = 4,
    ) -> IKResult:
        """Run several differential IK iterations from the current qpos."""
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if iterations < 1:
            raise ValueError("iterations must be at least 1.")

        current_qpos_array = np.asarray(current_qpos, dtype=float)
        if current_qpos_array.shape != (self.model.nq,):
            raise ValueError(
                f"qpos shape must be ({self.model.nq},), "
                f"got {current_qpos_array.shape}."
            )

        if not self._initialized:
            self.reset(current_qpos_array)
        else:
            # Closed-loop IK: start from the actual simulated configuration.
            self.configuration.update(current_qpos_array)

        # The iterations are IK substeps within one controller period.  Passing
        # the full controller dt to every iteration would make the requested
        # motion ``iterations`` times faster than max_joint_velocity.
        integration_dt = dt / iterations
        solution_found = True
        for _ in range(iterations):
            try:
                velocity = mink.solve_ik(
                    configuration=self.configuration,
                    tasks=[self.frame_task, self.posture_task],
                    dt=integration_dt,
                    solver=self.solver,
                    limits=self.limits,
                    damping=1e-6,
                )
            except mink.NoSolutionFound:
                # An unreachable target can make the constrained QP infeasible
                # at a joint boundary. Keep the last valid command rather than
                # terminating the simulator or commanding an invalid posture.
                solution_found = False
                break

            # MINK sees the complete scene, including the block freejoint and
            # gripper. Restrict the solution to the five arm hinge joints.
            arm_velocity = np.zeros(self.model.nv, dtype=float)
            arm_velocity[self.arm_dof_addresses] = velocity[self.arm_dof_addresses]

            self.configuration.integrate_inplace(arm_velocity, integration_dt)

        current_position = self.get_end_effector_position()
        error = float(np.linalg.norm(self._target_position - current_position))

        return IKResult(
            arm_qpos=np.clip(
                self.configuration.q[self.arm_qpos_addresses],
                self.arm_ctrl_ranges[:, 0],
                self.arm_ctrl_ranges[:, 1],
            ),
            position_error=error,
            current_position=current_position,
            target_position=self._target_position.copy(),
            solution_found=solution_found,
        )
