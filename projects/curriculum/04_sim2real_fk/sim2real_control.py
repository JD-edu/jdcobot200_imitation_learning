"""실제 JDCobot200과 MuJoCo를 안전하게 동기 제어하는 공통 코드."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Callable, Mapping

import mujoco
from mujoco.glfw import glfw
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COURSE03 = PROJECT_ROOT / "projects/curriculum/03_mujoco_fk_safety"
for path in (PROJECT_ROOT, COURSE03):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jdcobot200 import load_robot_config  # noqa: E402
from jdcobot200.hardware import JDCobot200Robot  # noqa: E402
from fk_safety import FKSafety, JOINT_NAMES  # noqa: E402


def resolve_joint(value: str) -> int:
    """CLI의 1~5 또는 관절 이름을 0-based index로 바꿉니다."""
    if value.isdigit() and 1 <= int(value) <= len(JOINT_NAMES):
        return int(value) - 1
    try:
        return JOINT_NAMES.index(value)
    except ValueError as exc:
        raise ValueError("관절은 1~5 또는 " + ", ".join(JOINT_NAMES) + " 중 하나여야 합니다.") from exc


def require_ticks(values: Mapping[str, int | None]) -> dict[str, int]:
    missing = [name for name in JOINT_NAMES if values.get(name) is None]
    if missing:
        raise RuntimeError(f"encoder 읽기 실패: {', '.join(missing)}")
    return {name: int(values[name]) for name in JOINT_NAMES}  # type: ignore[arg-type]


class MuJoCoViewer:
    """동기 제어용 최소 GLFW viewer."""

    def __init__(self, safety: FKSafety, title: str) -> None:
        self.safety = safety
        self.model = safety.model
        self.data = safety.data
        if not glfw.init():
            raise RuntimeError("GLFW 초기화 실패: DISPLAY 설정을 확인하세요.")
        self.window = glfw.create_window(900, 700, title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("GLFW 창 생성 실패")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10_000)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.12, 0.0, 0.12]
        self.camera.distance = 0.75
        self.camera.azimuth = 145.0
        self.camera.elevation = -25.0
        self.option = mujoco.MjvOption()
        glfw.set_key_callback(
            self.window,
            lambda window, key, scancode, action, mods:
            glfw.set_window_should_close(window, True)
            if action == glfw.PRESS and key == glfw.KEY_ESCAPE else None,
        )

    def set_key_callback(self, callback) -> None:
        glfw.set_key_callback(self.window, callback)

    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.window))

    def close(self) -> None:
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def render(self, q_rad: np.ndarray, title: str | None = None) -> None:
        self.data.qpos[self.safety.qpos_addresses] = q_rad
        mujoco.mj_forward(self.model, self.data)
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model, self.data, self.option, None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        if title:
            glfw.set_window_title(self.window, title)
        glfw.swap_buffers(self.window)
        glfw.poll_events()


class SynchronizedController:
    def __init__(self, *, port: str | None = None) -> None:
        self.config = load_robot_config()
        self.robot = JDCobot200Robot(self.config, port=port)
        self.safety = FKSafety()
        if tuple(j.name for j in self.config.arm_joints) != JOINT_NAMES:
            raise ValueError("robot.yaml과 MJCF의 arm 관절 순서가 다릅니다.")

    def close(self) -> None:
        self.robot.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def ticks_to_q(self, raw: Mapping[str, int]) -> np.ndarray:
        return np.asarray([
            self.robot.raw_to_radian(joint, raw[joint.name])
            for joint in self.config.arm_joints
        ])

    def q_to_ticks(self, q_rad: np.ndarray) -> dict[str, int]:
        result = {}
        for joint, angle in zip(self.config.arm_joints, np.asarray(q_rad, dtype=float)):
            tick = self.robot.radian_to_raw_tick(joint, float(angle))
            self.robot.validate_raw_target(joint, tick)
            result[joint.name] = tick
        return result

    def read(self, samples: int = 3) -> tuple[dict[str, int], np.ndarray]:
        raw = require_ticks(self.robot.read_raw_positions(samples=samples))
        return raw, self.ticks_to_q(raw)

    def check_path(self, start_q: np.ndarray, goal_q: np.ndarray, steps: int = 101):
        safe, index, result = self.safety.check_path(start_q, goal_q, steps)
        if not safe:
            raise RuntimeError(
                f"안전검사 거부: step={index}/{steps - 1}, "
                f"q(deg)={np.round(np.rad2deg(np.linspace(start_q, goal_q, steps)[index]), 2)}, "
                f"reason={result.reasons}"
            )
        return result

    def move_to(
        self, start_raw: Mapping[str, int], start_q: np.ndarray, goal_q: np.ndarray, *,
        duration: float = 2.0, rate_hz: float = 20.0,
        feedback_tolerance_ticks: int = 15, feedback_timeout: float = 2.0,
        on_step: Callable[[np.ndarray], None] | None = None,
    ) -> tuple[dict[str, int], np.ndarray]:
        """전체 경로를 먼저 검사하고 실제·MuJoCo에 같은 q trajectory를 적용합니다."""
        if (duration <= 0 or rate_hz <= 0 or feedback_tolerance_ticks < 0
                or feedback_timeout <= 0):
            raise ValueError(
                "duration, rate_hz, feedback_timeout은 양수이고 "
                "feedback_tolerance_ticks는 0 이상이어야 합니다."
            )
        self.check_path(start_q, goal_q, max(101, int(duration * rate_hz) + 1))
        goal_raw = self.q_to_ticks(goal_q)
        self.robot.hold_and_enable(start_raw, self.config.arm_joints)

        steps = max(1, int(round(duration * rate_hz)))
        period = duration / steps
        for index in range(1, steps + 1):
            progress = index / steps
            alpha = progress * progress * (3.0 - 2.0 * progress)
            command_q = (1.0 - alpha) * start_q + alpha * goal_q
            # 사전검사 후에도 실제 전송 직전 각 명령을 다시 검사합니다.
            command_result = self.safety.evaluate(command_q)
            if not command_result.safe:
                raise RuntimeError(f"이동 중 안전검사 거부: {command_result.reasons}")
            command_raw = self.q_to_ticks(command_q)
            for joint in self.config.arm_joints:
                self.robot.driver.set_position(joint.motor_id, command_raw[joint.name])
                time.sleep(0.002)
            if on_step:
                on_step(command_q)
            time.sleep(period)

        measured = self.robot.wait_for_raw_targets(
            goal_raw,
            tolerance=feedback_tolerance_ticks,
            timeout=feedback_timeout,
        )
        measured_q = self.ticks_to_q(measured)
        measured_result = self.safety.evaluate(measured_q)
        if not measured_result.safe:
            raise RuntimeError(f"도착 encoder 자세가 안전하지 않습니다: {measured_result.reasons}")
        return measured, measured_q
