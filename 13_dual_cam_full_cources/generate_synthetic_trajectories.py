#!/usr/bin/env python3
"""Generate diverse JDCobot200 pick-and-place demonstration trajectories.

The output is deliberately framework-neutral.  Each episode is a compressed
NumPy file containing synchronized observations, actions, task state, and the
Cartesian reference followed by the expert.  It can later be converted to a
LeRobot dataset without rerunning MuJoCo.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from jdcobot200_mink_solver import SO101MinkSolver


ROOT = Path(__file__).resolve().parent
LOCAL_XML = ROOT / "scene.xml"
DEFAULT_XML = LOCAL_XML
ARM_NAMES = SO101MinkSolver.ARM_JOINT_NAMES
GOAL_XY = np.array([0.22, 0.10])
BLOCK_Z = 0.016
GRASP_Z = 0.018
OPEN = 0.45
CLOSED = -0.20
WRIST_START = 2.14746


@dataclass(frozen=True)
class Segment:
    name: str
    target: np.ndarray
    gripper: float
    duration: float
    grasp_at_start: bool = False
    release_at_start: bool = False


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"Missing MuJoCo object: {name}")
    return int(value)


def minimum_jerk(x: float) -> float:
    """Zero-velocity/acceleration interpolation in [0, 1]."""
    x = float(np.clip(x, 0.0, 1.0))
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


class TrajectoryGenerator:
    def __init__(
        self,
        xml: Path,
        control_hz: int,
        min_floor_clearance: float,
        self_collision_margin: float,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml.resolve()))
        self.dt = 1.0 / control_hz
        self.min_floor_clearance = float(min_floor_clearance)
        self.self_collision_margin = float(self_collision_margin)
        self.substeps = max(1, round(self.dt / self.model.opt.timestep))
        self.arm_act = np.array([
            object_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ARM_NAMES
        ])
        joints = np.array([
            object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_NAMES
        ])
        self.arm_qadr = self.model.jnt_qposadr[joints]
        self.wrist_index = ARM_NAMES.index("wrist_roll")
        self.gripper_act = object_id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor"
        )
        self.gripper_joint = object_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_left"
        )
        self.gripper_qadr = int(self.model.jnt_qposadr[self.gripper_joint])
        self.tool_site = object_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "graspframe"
        )
        self.gripper_body = object_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_assembly"
        )
        self.block_body = object_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block"
        )
        self.block_joint = object_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_freejoint"
        )
        self.block_qadr = int(self.model.jnt_qposadr[self.block_joint])
        self.grasp_weld = object_id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "red_block_grasp"
        )
        self.pad_geoms = np.array([
            object_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("left_finger_pad", "right_finger_pad")
        ])
        floor_id = object_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        base_body = object_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_assembly"
        )
        # Validate every moving robot geom, including visual STL meshes.  This
        # works even though their MuJoCo contact masks are intentionally off.
        self.floor_z = float(self.model.geom_pos[floor_id, 2])
        self.robot_geoms = np.array([
            geom_id for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id])
            not in (0, base_body, self.block_body)
        ])
        self.mesh_vertices: dict[int, np.ndarray] = {}
        for geom_id in self.robot_geoms:
            if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = int(self.model.geom_dataid[geom_id])
            start = int(self.model.mesh_vertadr[mesh_id])
            count = int(self.model.mesh_vertnum[mesh_id])
            self.mesh_vertices[int(geom_id)] = self.model.mesh_vert[
                start:start + count
            ].copy()
        # Geometries belonging to the same rigid body or two directly joined
        # bodies overlap by design around a joint. Only non-adjacent links are
        # meaningful self-collision candidates.
        self.self_collision_pairs: list[tuple[int, int]] = []
        def are_nearby_bodies(body_a: int, body_b: int) -> bool:
            """True for the same, parent/child, or sibling mechanism bodies."""
            if body_a == body_b:
                return True
            parent_a = int(self.model.body_parentid[body_a])
            parent_b = int(self.model.body_parentid[body_b])
            return parent_a == body_b or parent_b == body_a or parent_a == parent_b

        all_robot_geoms = [
            geom_id for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) not in (0, self.block_body)
        ]
        for offset, geom_a in enumerate(all_robot_geoms):
            body_a = int(self.model.geom_bodyid[geom_a])
            for geom_b in all_robot_geoms[offset + 1:]:
                body_b = int(self.model.geom_bodyid[geom_b])
                if are_nearby_bodies(body_a, body_b):
                    continue
                self.self_collision_pairs.append((geom_a, geom_b))

    def sample_initial_arm(self, rng: np.random.Generator) -> np.ndarray:
        """Random, conservative joint pose inside the calibrated ctrl ranges."""
        ranges = self.model.actuator_ctrlrange[self.arm_act]
        # Using most of base range and a conservative envelope for the planar
        # joints avoids starting below the table or in a singular folded pose.
        centers = np.array([0.0, -0.20, 0.75, 0.15, WRIST_START])
        widths = np.array([1.20, 0.32, 0.55, 0.16, 0.30])
        q = centers + rng.uniform(-widths, widths)
        return np.clip(q, ranges[:, 0] + 0.05, ranges[:, 1] - 0.05)

    @staticmethod
    def sample_block_xy(rng: np.random.Generator) -> np.ndarray:
        # Front and side workspace, kept clear of the fixed destination.
        for _ in range(100):
            xy = np.array([rng.uniform(0.17, 0.275), rng.uniform(-0.15, 0.015)])
            if np.linalg.norm(xy - GOAL_XY) > 0.09:
                return xy
        raise RuntimeError("Could not sample a valid block position")

    def make_segments(
        self, rng: np.random.Generator, start_xyz: np.ndarray, pick_xy: np.ndarray
    ) -> list[Segment]:
        pick_hover = rng.uniform(0.135, 0.18)
        carry_z = rng.uniform(0.15, 0.20)
        speed = rng.uniform(0.88, 1.15)
        # Two different randomized Bézier-like bends make demonstrations
        # multimodal while retaining safe vertical approach and release.
        pre_pick = np.array([
            rng.uniform(0.15, 0.27), rng.uniform(-0.18, 0.08),
            rng.uniform(0.14, 0.21),
        ])
        midpoint = (pick_xy + GOAL_XY) / 2
        carry_1 = np.r_[
            midpoint + rng.uniform([-0.045, -0.07], [0.045, 0.07]), carry_z
        ]
        goal_hover_z = rng.uniform(0.14, 0.18)
        return [
            Segment("start", start_xyz.copy(), OPEN, 0.20),
            Segment("pre_pick_waypoint", pre_pick, OPEN, 0.65 * speed),
            Segment("pick_hover", np.r_[pick_xy, pick_hover], OPEN, 0.65 * speed),
            Segment("pick_descend", np.r_[pick_xy, GRASP_Z], OPEN, 0.65 * speed),
            Segment("grasp_close", np.r_[pick_xy, GRASP_Z], CLOSED, 0.45),
            Segment(
                "lift", np.r_[pick_xy, carry_z], CLOSED, 0.65 * speed,
                grasp_at_start=True,
            ),
            Segment("carry_waypoint", carry_1, CLOSED, 0.70 * speed),
            Segment(
                "goal_hover", np.r_[GOAL_XY, goal_hover_z], CLOSED,
                0.70 * speed,
            ),
            Segment("goal_align_1", np.r_[GOAL_XY, goal_hover_z], CLOSED, 0.45),
            Segment("goal_align_2", np.r_[GOAL_XY, goal_hover_z], CLOSED, 0.45),
            Segment("place_descend", np.r_[GOAL_XY, GRASP_Z], CLOSED, 0.90 * speed),
            Segment("place_correct_1", np.r_[GOAL_XY, GRASP_Z], CLOSED, 0.45),
            Segment("place_correct_2", np.r_[GOAL_XY, GRASP_Z], CLOSED, 0.45),
            Segment(
                "release", np.r_[GOAL_XY, GRASP_Z], OPEN, 0.40,
                release_at_start=True,
            ),
            Segment("retreat", np.r_[GOAL_XY, goal_hover_z], OPEN, 0.65 * speed),
        ]

    def activate_weld(self, data: mujoco.MjData) -> None:
        gripper_pos = data.xpos[self.gripper_body]
        gripper_mat = data.xmat[self.gripper_body].reshape(3, 3)
        block_pos = data.xpos[self.block_body]
        inverse_gripper_quat = data.xquat[self.gripper_body].copy()
        inverse_gripper_quat[1:] *= -1
        relative_quat = np.empty(4)
        mujoco.mju_mulQuat(
            relative_quat, inverse_gripper_quat, data.xquat[self.block_body]
        )
        eq = self.model.eq_data[self.grasp_weld]
        eq[:3] = 0.0
        eq[3:6] = gripper_mat.T @ (block_pos - gripper_pos)
        eq[6:10] = relative_quat
        data.eq_active[self.grasp_weld] = 1

    def aligned_wrist(self, data: mujoco.MjData) -> float:
        line = data.geom_xpos[self.pad_geoms[1]] - data.geom_xpos[self.pad_geoms[0]]
        yaw = np.arctan2(line[1], line[0])
        error = np.arctan2(np.sin(-np.pi / 2 - yaw), np.cos(-np.pi / 2 - yaw))
        current = data.qpos[self.arm_qadr[self.wrist_index]]
        lo, hi = self.model.actuator_ctrlrange[self.arm_act[self.wrist_index]]
        return float(np.clip(current + error, lo, hi))

    def floor_clearance(self, data: mujoco.MjData) -> tuple[float, int]:
        """Return the lowest moving-robot point above the floor and its geom."""
        lowest = np.inf
        lowest_geom = -1
        for geom_id_raw in self.robot_geoms:
            geom_id = int(geom_id_raw)
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            position = data.geom_xpos[geom_id]
            if geom_id in self.mesh_vertices:
                # Only the world-Z coordinate is required.
                geom_low = float(np.min(
                    self.mesh_vertices[geom_id] @ rotation[2] + position[2]
                ))
            elif self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                geom_low = float(
                    position[2]
                    - np.abs(rotation[2]) @ self.model.geom_size[geom_id]
                )
            else:
                # Current moving robot geoms are meshes and fingertip boxes.
                # This conservative fallback uses the bounding radius.
                geom_low = float(position[2] - self.model.geom_rbound[geom_id])
            if geom_low < lowest:
                lowest = geom_low
                lowest_geom = geom_id
        return lowest - self.floor_z, lowest_geom

    def geom_label(self, geom_id: int) -> str:
        geom_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        body_id = int(self.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        return geom_name or f"{body_name}/geom_{geom_id}"

    def self_collision(self, data: mujoco.MjData) -> tuple[bool, float, tuple[int, int]]:
        """Check exact MuJoCo distances between non-adjacent robot links."""
        closest = np.inf
        closest_pair = (-1, -1)
        fromto = np.empty(6, dtype=float)
        # A short cutoff keeps per-frame checking cheap. Distances reported as
        # 0.02 m mean "at least 0.02 m"; penetrations remain exact and negative.
        distance_query_limit = max(0.02, self.self_collision_margin)
        for geom_a, geom_b in self.self_collision_pairs:
            distance = float(mujoco.mj_geomDistance(
                self.model,
                data,
                geom_a,
                geom_b,
                distance_query_limit,
                fromto,
            ))
            if distance < closest:
                closest = distance
                closest_pair = (geom_a, geom_b)
            if distance < self.self_collision_margin:
                return True, distance, closest_pair
        return False, closest, closest_pair

    def generate(self, episode_index: int, seed: int) -> tuple[dict[str, np.ndarray], dict]:
        rng = np.random.default_rng(seed)
        data = mujoco.MjData(self.model)
        data.eq_active[self.grasp_weld] = 0
        initial_arm = self.sample_initial_arm(rng)
        data.qpos[self.arm_qadr] = initial_arm
        data.qpos[self.gripper_qadr] = OPEN
        pick_xy = self.sample_block_xy(rng)
        data.qpos[self.block_qadr:self.block_qadr + 7] = np.r_[
            pick_xy, BLOCK_Z, 1.0, 0.0, 0.0, 0.0
        ]
        data.ctrl[self.arm_act] = initial_arm
        data.ctrl[self.gripper_act] = OPEN
        mujoco.mj_forward(self.model, data)

        start_xyz = data.site_xpos[self.tool_site].copy()
        # If a sampled pose puts the tool too low, begin with a safe Cartesian
        # reference; the recorded state is still the genuinely random qpos.
        segments = self.make_segments(rng, start_xyz, pick_xy)
        solver = SO101MinkSolver(self.model, end_effector_site="graspframe")
        solver.reset(data.qpos.copy())

        rows: dict[str, list[np.ndarray | float | int]] = {
            "timestamp": [], "qpos": [], "qvel": [], "action": [],
            "ee_position": [], "block_position": [], "block_quaternion": [],
            "desired_ee_position": [], "phase": [],
        }
        previous_target = start_xyz.copy()
        minimum_clearance = np.inf
        minimum_clearance_frame = -1
        minimum_clearance_geom = -1
        finite_values = True
        joint_limits_valid = True
        self_collision_valid = True
        minimum_self_distance = np.inf
        self_collision_frame = -1
        self_collision_pair = (-1, -1)
        for phase_id, segment in enumerate(segments):
            if segment.grasp_at_start:
                self.activate_weld(data)
            if segment.release_at_start:
                data.eq_active[self.grasp_weld] = 0
            segment_target = segment.target.copy()
            if segment.name.startswith("goal_align"):
                desired_block = data.xpos[self.block_body].copy()
                desired_block[:2] = GOAL_XY
                segment_target = (
                    data.site_xpos[self.tool_site]
                    + desired_block
                    - data.xpos[self.block_body]
                )
            if segment.name.startswith("place_") and segment.name != "place_marker":
                # The weld preserves the actual grasp offset, which varies a
                # little with controller tracking.  Compensate that measured
                # offset so it is the block (not merely the tool reference)
                # that arrives at the fixed destination.
                desired_block = np.r_[GOAL_XY, BLOCK_Z]
                segment_target = (
                    data.site_xpos[self.tool_site]
                    + desired_block
                    - data.xpos[self.block_body]
                )
            steps = max(2, round(segment.duration / self.dt))
            for step in range(steps):
                alpha = minimum_jerk((step + 1) / steps)
                target = previous_target + alpha * (segment_target - previous_target)
                solver.set_target_position(target)
                result = solver.solve_step(data.qpos.copy(), self.dt, iterations=5)
                command = result.arm_qpos.copy()
                command[self.wrist_index] = self.aligned_wrist(data)
                data.ctrl[self.arm_act] = command
                data.ctrl[self.gripper_act] = segment.gripper
                for _ in range(self.substeps):
                    mujoco.mj_step(self.model, data)

                clearance, clearance_geom = self.floor_clearance(data)
                if clearance < minimum_clearance:
                    minimum_clearance = clearance
                    minimum_clearance_frame = len(rows["timestamp"])
                    minimum_clearance_geom = clearance_geom
                colliding, self_distance, collision_pair = self.self_collision(data)
                if self_distance < minimum_self_distance:
                    minimum_self_distance = self_distance
                    self_collision_frame = len(rows["timestamp"])
                    self_collision_pair = collision_pair
                self_collision_valid = self_collision_valid and not colliding
                finite_values = finite_values and bool(
                    np.all(np.isfinite(data.qpos))
                    and np.all(np.isfinite(data.qvel))
                    and np.all(np.isfinite(command))
                )
                arm_actual = data.qpos[self.arm_qadr]
                ranges = self.model.actuator_ctrlrange[self.arm_act]
                joint_limits_valid = joint_limits_valid and bool(
                    np.all(arm_actual >= ranges[:, 0] - 0.02)
                    and np.all(arm_actual <= ranges[:, 1] + 0.02)
                )

                rows["timestamp"].append(float(data.time))
                rows["qpos"].append(data.qpos.copy())
                rows["qvel"].append(data.qvel.copy())
                rows["action"].append(np.r_[command, segment.gripper])
                rows["ee_position"].append(data.site_xpos[self.tool_site].copy())
                rows["block_position"].append(data.xpos[self.block_body].copy())
                rows["block_quaternion"].append(data.xquat[self.block_body].copy())
                rows["desired_ee_position"].append(target.copy())
                rows["phase"].append(phase_id)
            previous_target = segment_target.copy()

        arrays = {key: np.asarray(value) for key, value in rows.items()}
        final_error = float(np.linalg.norm(arrays["block_position"][-1, :2] - GOAL_XY))
        placement_valid = final_error < 0.035
        floor_valid = minimum_clearance >= self.min_floor_clearance
        rejection_reasons = []
        if not placement_valid:
            rejection_reasons.append("placement_error")
        if not floor_valid:
            rejection_reasons.append("floor_penetration")
        if not self_collision_valid:
            rejection_reasons.append("self_collision")
        if not joint_limits_valid:
            rejection_reasons.append("joint_limit")
        if not finite_values:
            rejection_reasons.append("non_finite")
        metadata = {
            "episode_index": episode_index,
            "seed": seed,
            "length": len(arrays["timestamp"]),
            "duration_s": float(arrays["timestamp"][-1]),
            "initial_arm_qpos": initial_arm.tolist(),
            "initial_block_xy": pick_xy.tolist(),
            "goal_xy": GOAL_XY.tolist(),
            "final_block_xyz": arrays["block_position"][-1].tolist(),
            "final_place_error_m": final_error,
            "floor_clearance_min_m": float(minimum_clearance),
            "floor_clearance_frame": minimum_clearance_frame,
            "floor_clearance_geom": self.geom_label(minimum_clearance_geom),
            "floor_clearance_required_m": self.min_floor_clearance,
            "floor_valid": floor_valid,
            "self_collision_valid": self_collision_valid,
            "self_collision_margin_m": self.self_collision_margin,
            "minimum_self_distance_m": float(minimum_self_distance),
            "self_collision_frame": self_collision_frame,
            "closest_self_collision_geoms": [
                self.geom_label(geom_id) for geom_id in self_collision_pair
                if geom_id >= 0
            ],
            "joint_limits_valid": joint_limits_valid,
            "finite_values": finite_values,
            "placement_valid": placement_valid,
            "rejection_reasons": rejection_reasons,
            "success": not rejection_reasons,
            "phase_names": [s.name for s in segments],
        }
        return arrays, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "synthetic_dataset")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=25)
    parser.add_argument(
        "--candidate-multiplier", type=int, default=2,
        help="초기 후보 예산 배수(부족하면 성공 개수까지 계속 생성)",
    )
    parser.add_argument(
        "--min-floor-clearance", type=float, default=0.0,
        help="움직이는 모든 로봇 형상에 요구할 최소 바닥 여유(m)",
    )
    parser.add_argument(
        "--self-collision-margin", type=float, default=0.0,
        help="인접하지 않은 로봇 링크 사이에 요구할 최소 여유(m)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.control_hz < 1:
        raise ValueError("episodes and control-hz must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.glob("episode_*.npz"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already contains episodes; pass --overwrite"
        )
    if args.overwrite:
        for episode_file in existing:
            episode_file.unlink()
        manifest_file = args.output_dir / "manifest.json"
        if manifest_file.exists():
            manifest_file.unlink()

    if args.max_attempts < 1 or args.candidate_multiplier < 1:
        raise ValueError("max-attempts와 candidate-multiplier는 양수여야 합니다")
    generator = TrajectoryGenerator(
        args.xml,
        args.control_hz,
        args.min_floor_clearance,
        args.self_collision_margin,
    )
    manifest = {
        "format_version": 1,
        "generator": Path(__file__).name,
        "xml": str(args.xml.resolve()),
        "control_hz": args.control_hz,
        "action_names": [*ARM_NAMES, "gripper"],
        "goal_xy": GOAL_XY.tolist(),
        "initial_candidate_budget": args.episodes * args.candidate_multiplier,
        "validation": {
            "headless_every_frame": True,
            "min_floor_clearance_m": args.min_floor_clearance,
            "moving_robot_mesh_vertices_checked": True,
            "self_collision_checked": True,
            "self_collision_margin_m": args.self_collision_margin,
            "joint_limits_checked": True,
            "finite_values_checked": True,
            "placement_checked": True,
        },
        "episodes": [],
    }
    successes = 0
    rejected = 0
    rejection_counts: dict[str, int] = {}
    for index in range(args.episodes):
        arrays = None
        metadata = None
        for attempt in range(1, args.max_attempts + 1):
            episode_seed = int(
                np.random.SeedSequence([args.seed, index, attempt]).generate_state(1)[0]
            )
            candidate_arrays, candidate_metadata = generator.generate(
                index, episode_seed
            )
            if candidate_metadata["success"]:
                arrays, metadata = candidate_arrays, candidate_metadata
                metadata["generation_attempt"] = attempt
                break
            rejected += 1
            for reason in candidate_metadata["rejection_reasons"]:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        if arrays is None or metadata is None:
            raise RuntimeError(
                f"Could not generate successful episode {index} in "
                f"{args.max_attempts} attempts"
            )
        filename = f"episode_{index:04d}.npz"
        np.savez_compressed(args.output_dir / filename, **arrays)
        metadata["file"] = filename
        manifest["episodes"].append(metadata)
        successes += int(metadata["success"])
        print(
            f"[{index + 1:03d}/{args.episodes:03d}] {filename} "
            f"block={np.round(metadata['initial_block_xy'], 3)} "
            f"error={metadata['final_place_error_m'] * 1000:.1f} mm "
            f"attempt={metadata['generation_attempt']}"
        )

    manifest["success_count"] = successes
    manifest["rejected_attempt_count"] = rejected
    manifest["total_candidate_count"] = args.episodes + rejected
    manifest["rejection_reason_counts"] = rejection_counts
    manifest["episode_count"] = args.episodes
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.episodes} trajectories to {args.output_dir}")
    if successes != args.episodes:
        raise RuntimeError(f"Only {successes}/{args.episodes} episodes passed validation")


if __name__ == "__main__":
    main()
