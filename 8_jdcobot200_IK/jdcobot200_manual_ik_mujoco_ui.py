#!/usr/bin/env python3
"""Manual real-robot IK UI with synchronized MuJoCo comparison.

The existing manual UI is reused without modification:
  - left MuJoCo viewport: commanded IK joint pose
  - right MuJoCo viewport: measured real-robot encoder pose

This is a separate diagnostic entry point. Closing either window or pressing
ESC in either window releases torque and closes the application.
"""

from __future__ import annotations

import sys
import time
import tkinter as tk

import mujoco
import numpy as np
from mujoco.glfw import glfw

from jdcobot200_manual_ik_ui import ManualIKUI, parse_args


class ManualIKMuJoCoUI(ManualIKUI):
    def __init__(self, root: tk.Tk, args) -> None:
        self.simulation_ready = False
        super().__init__(root, args)
        try:
            self._create_mujoco_window()
            self.simulation_ready = True
        except Exception:
            super().close()
            raise

    def _create_mujoco_window(self) -> None:
        if not glfw.init():
            raise RuntimeError(
                "GLFW 초기화 실패: DISPLAY와 그래픽 드라이버를 확인하세요."
            )
        self.mujoco_window = glfw.create_window(
            1400, 760, "JDCobot200 IK command vs real encoder", None, None
        )
        if self.mujoco_window is None:
            glfw.terminate()
            raise RuntimeError("MuJoCo GLFW 창 생성 실패")
        glfw.make_context_current(self.mujoco_window)
        glfw.swap_interval(1)

        model = self.safety.model
        self.command_data = mujoco.MjData(model)
        self.measured_data = mujoco.MjData(model)
        self.scene = mujoco.MjvScene(model, maxgeom=10_000)
        self.context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.25]
        self.camera.distance = 0.75
        self.camera.azimuth = 145.0
        self.camera.elevation = -25.0
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.option)

        def on_key(window, key, scancode, action, mods):
            if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
                self.emergency_stop()
                glfw.set_window_should_close(window, True)

        glfw.set_key_callback(self.mujoco_window, on_key)

    def _set_simulation_pose(self, data: mujoco.MjData, q: np.ndarray) -> None:
        data.qpos[:] = 0.0
        data.qpos[self.safety.qpos_addresses] = np.clip(
            q, self.safety.lower, self.safety.upper
        )
        mujoco.mj_forward(self.safety.model, data)

    def _render_view(
        self, data: mujoco.MjData, viewport: mujoco.MjrRect, label: str,
        detail: str,
    ) -> None:
        mujoco.mjv_updateScene(
            self.safety.model,
            data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            viewport,
            label,
            detail,
            self.context,
        )

    def _render_mujoco(self) -> None:
        if not self.simulation_ready:
            return
        glfw.make_context_current(self.mujoco_window)
        glfw.poll_events()
        if glfw.window_should_close(self.mujoco_window):
            self.close()
            return

        with self.feedback_lock:
            measured_q = self.measured_q.copy()
            feedback_age = time.perf_counter() - self.feedback_last_ok
        command_q = self.command_q.copy()
        self._set_simulation_pose(self.command_data, command_q)
        self._set_simulation_pose(self.measured_data, measured_q)

        command_tcp = (
            self.command_data.site_xpos[self.safety.site_id].copy() * 1000.0
        )
        measured_tcp = (
            self.measured_data.site_xpos[self.safety.site_id].copy() * 1000.0
        )
        joint_error = np.rad2deg(command_q - measured_q)
        tcp_error = command_tcp - measured_tcp

        width, height = glfw.get_framebuffer_size(self.mujoco_window)
        half = width // 2
        left = mujoco.MjrRect(0, 0, half, height)
        right = mujoco.MjrRect(half, 0, width - half, height)
        self._render_view(
            self.command_data,
            left,
            "COMMAND / IK",
            f"TCP mm: {np.round(command_tcp, 1)}",
        )
        self._render_view(
            self.measured_data,
            right,
            "MEASURED / REAL ENCODER",
            f"TCP mm: {np.round(measured_tcp, 1)}",
        )
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value,
            right,
            f"max joint error: {np.max(np.abs(joint_error)):.2f} deg",
            f"TCP difference: {np.linalg.norm(tcp_error):.2f} mm | "
            f"feedback age: {feedback_age:.2f} s",
            self.context,
        )
        glfw.swap_buffers(self.mujoco_window)

    def _update(self) -> None:
        if self.closed:
            return
        # Parent handles serial commands, background feedback, completion
        # detection, and scheduling the next call to this overridden method.
        super()._update()
        if not self.closed:
            self._render_mujoco()

    def close(self) -> None:
        if self.closed:
            return
        if self.simulation_ready:
            self.simulation_ready = False
            try:
                glfw.make_context_current(self.mujoco_window)
                self.context.free()
                glfw.destroy_window(self.mujoco_window)
            finally:
                glfw.terminate()
        super().close()


def main() -> int:
    args = parse_args()
    if min(
        args.cartesian_step_mm,
        args.joint_step_deg,
        args.max_joint_speed_deg_s,
        args.encoder_period,
    ) <= 0:
        raise ValueError("step, speed, encoder-period는 양수여야 합니다.")
    if args.joint_step_deg > 5.0 or args.max_joint_speed_deg_s > 15.0:
        raise ValueError("수동 joint step은 5°, 속도는 15°/s 이하로 설정하세요.")
    root = tk.Tk()
    ManualIKMuJoCoUI(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
