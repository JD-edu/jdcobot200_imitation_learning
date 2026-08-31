#!/usr/bin/env python3
"""SO-101 MuJoCo simulation using GLFW and a separate MINK solver module.

Terminal commands while the GLFW window is running:
    x y z           Move graspframe to the world-coordinate target in metres.
    home            Return to the initial graspframe position.
    gripper open    Open the gripper.
    gripper close   Close the gripper.
    quit            Close the program.

GLFW controls:
    Left drag       Orbit camera
    Right drag      Pan camera
    Middle drag     Zoom camera
    Mouse wheel     Zoom camera
    Esc             Close window
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

from mujoco.glfw import glfw
import mujoco
import numpy as np

from jdcobot200_mink_solver import SO101MinkSolver


XML_PATH = Path(__file__).resolve().parent / "scene.xml"

CONTROL_HZ = 100.0
CONTROL_DT = 1.0 / CONTROL_HZ
IK_ITERATIONS = 5

END_EFFECTOR_SITE = "graspframe"
#BLOCK_SITE = "block_center"

ARM_JOINT_NAMES = SO101MinkSolver.ARM_JOINT_NAMES
GRIPPER_ACTUATOR_NAME = "gripper_motor"

GRIPPER_OPEN = 1.20
GRIPPER_CLOSED = 0.00


class GLFWCameraController:
    def __init__(self, model: mujoco.MjModel, scene: mujoco.MjvScene) -> None:
        self.model = model
        self.scene = scene
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()

        mujoco.mjv_defaultCamera(self.camera)
        mujoco.mjv_defaultOption(self.option)

        self.camera.azimuth = 160.0
        self.camera.elevation = -20.0
        self.camera.distance = 1.2
        self.camera.lookat[:] = np.array([0.20, 0.0, 0.25])

        self._button_left = False
        self._button_middle = False
        self._button_right = False
        self._last_x = 0.0
        self._last_y = 0.0

    def mouse_button(self, window, button, action, mods) -> None:
        self._button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self._button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self._last_x, self._last_y = glfw.get_cursor_pos(window)

    def cursor_position(self, window, xpos, ypos) -> None:
        if not (self._button_left or self._button_middle or self._button_right):
            return

        dx = xpos - self._last_x
        dy = ypos - self._last_y
        self._last_x = xpos
        self._last_y = ypos

        width, height = glfw.get_window_size(window)
        if height <= 0:
            return

        shift_pressed = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if self._button_right:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift_pressed
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self._button_left:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift_pressed
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(
            self.model,
            action,
            dx / height,
            dy / height,
            self.scene,
            self.camera,
        )

    def scroll(self, window, xoffset, yoffset) -> None:
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene,
            self.camera,
        )


def require_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        raise ValueError(f"MuJoCo model does not contain '{name}'.")
    return int(object_id)


def input_worker(command_queue: queue.Queue[str], stop_event: threading.Event) -> None:
    print(
        "\n명령 입력: x y z | home | "
        "gripper open/close | quit"
    )
    while not stop_event.is_set():
        try:
            line = input("target> ").strip()
        except EOFError:
            command_queue.put("quit")
            return
        except KeyboardInterrupt:
            command_queue.put("quit")
            return

        if line:
            command_queue.put(line)


def parse_xyz(parts: list[str]) -> np.ndarray:
    if len(parts) != 3:
        raise ValueError("좌표는 x y z 세 값을 입력해야 합니다.")
    xyz = np.array([float(value) for value in parts], dtype=float)
    if not np.all(np.isfinite(xyz)):
        raise ValueError("좌표는 유한한 숫자여야 합니다.")
    return xyz


def main() -> None:
    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"scene.xml을 찾을 수 없습니다: {XML_PATH}\n"
            "scene.xml, so101_new_calib.xml, meshes/를 같은 프로젝트에 배치하세요."
        )

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    arm_joint_ids = np.array(
        [require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES],
        dtype=int,
    )
    arm_qpos_addresses = model.jnt_qposadr[arm_joint_ids].astype(int)

    # XML actuator names are identical to the five arm joint names.
    arm_actuator_ids = np.array(
        [require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINT_NAMES],
        dtype=int,
    )
    gripper_actuator_id = require_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
    )

    end_effector_site_id = require_id(
        model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE
    )
    # Start actuators at the model's initial joint configuration.
    mujoco.mj_forward(model, data)
    data.ctrl[arm_actuator_ids] = data.qpos[arm_qpos_addresses]
    data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED

    solver = SO101MinkSolver(model=model, end_effector_site=END_EFFECTOR_SITE)
    solver.reset(data.qpos.copy())

    home_target = data.site_xpos[end_effector_site_id].copy()
    # Holding the initial pose is a safe startup behavior.  The former default
    # [0, 0, 0.06] lies near the base and is outside the arm's practical
    # workspace, causing the joints to slam into their limits immediately.
    current_target = home_target.copy()
    solver.set_target_position(current_target)

    print(f"초기 graspframe 위치: {home_target}")
    print(f"초기 목표 위치:       {current_target}")

    if not glfw.init():
        raise RuntimeError("GLFW initialization failed.")

    window = glfw.create_window(1200, 900, "SO-101 MuJoCo + MINK", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    camera_controller = GLFWCameraController(model, scene)

    glfw.set_mouse_button_callback(window, camera_controller.mouse_button)
    glfw.set_cursor_pos_callback(window, camera_controller.cursor_position)
    glfw.set_scroll_callback(window, camera_controller.scroll)

    def key_callback(window, key, scancode, action, mods) -> None:
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, key_callback)

    command_queue: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    input_thread = threading.Thread(
        target=input_worker,
        args=(command_queue, stop_event),
        daemon=True,
    )
    input_thread.start()

    next_control_time = time.perf_counter()
    last_status_time = 0.0
    latest_error = float("inf")
    warned_unreachable = False

    try:
        while not glfw.window_should_close(window):
            # Process all terminal commands without stopping the GLFW loop.
            while True:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    break

                parts = command.lower().split()
                try:
                    if len(parts) == 3:
                        current_target = parse_xyz(parts)
                        solver.set_target_position(current_target)
                        print(f"새 목표 위치: {current_target}")
                  
                    elif parts == ["home"]:
                        current_target = home_target.copy()
                        solver.set_target_position(current_target)
                        print(f"홈 목표 위치: {current_target}")


                    elif parts == ["gripper", "open"]:
                        data.ctrl[gripper_actuator_id] = GRIPPER_OPEN
                        print("그리퍼 열기")

                    elif parts == ["gripper", "close"]:
                        data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED
                        print("그리퍼 닫기")

                    elif parts == ["quit"]:
                        glfw.set_window_should_close(window, True)

                    else:
                        print(
                            "사용법: x y z | home | "
                            "gripper open | gripper close | quit"
                        )
                except ValueError as exc:
                    print(f"입력 오류: {exc}")

            now = time.perf_counter()
            while now >= next_control_time:
                result = solver.solve_step(
                    current_qpos=data.qpos.copy(),
                    dt=CONTROL_DT,
                    iterations=IK_ITERATIONS,
                )

                # MINK generates desired arm joint positions. MuJoCo position
                # actuators make the physical model track those positions.
                data.ctrl[arm_actuator_ids] = result.arm_qpos
                # CONTROL_DT is 10 ms while the XML timestep is 2 ms. Advance
                # all physics substeps so simulation time tracks wall time.
                substeps = max(1, round(CONTROL_DT / model.opt.timestep))
                for _ in range(substeps):
                    mujoco.mj_step(model, data)

                latest_error = result.position_error
                if not result.solution_found and not warned_unreachable:
                    print(
                        "IK 경고: 현재 목표에 대해 제한을 만족하는 해가 없습니다. "
                        "마지막 안전 자세를 유지합니다."
                    )
                    warned_unreachable = True
                elif result.solution_found:
                    warned_unreachable = False
                next_control_time += CONTROL_DT
                now = time.perf_counter()

                # Prevent an excessive catch-up loop after window dragging.
                if now - next_control_time > 0.25:
                    next_control_time = now
                    break

            if time.perf_counter() - last_status_time >= 0.5:
                actual_position = data.site_xpos[end_effector_site_id].copy()
                actual_error = np.linalg.norm(current_target - actual_position)
                glfw.set_window_title(
                    window,
                    "SO-101 MuJoCo + MINK | "
                    f"target=({current_target[0]:.3f}, "
                    f"{current_target[1]:.3f}, {current_target[2]:.3f}) | "
                    f"error={actual_error * 1000.0:.1f} mm",
                )
                last_status_time = time.perf_counter()

            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)

            mujoco.mjv_updateScene(
                model,
                data,
                camera_controller.option,
                None,
                camera_controller.camera,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                scene,
            )
            mujoco.mjr_render(viewport, scene, context)

            glfw.swap_buffers(window)
            glfw.poll_events()

    finally:
        stop_event.set()
        context.free()
        scene.free()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise
