#!/usr/bin/env python3
"""Slow manual Cartesian/joint jogging UI for the real JDCobot200.

This diagnostic tool intentionally does not apply the collision, floor,
Jacobian-singularity, or per-move IK-change checks used by the automatic
controllers. Hardware tick limits, a low joint-speed limit, explicit arming,
and ESC torque-off remain active.

Keyboard (window must have focus):
  A/D: -X/+X       S/W: -Y/+Y       F/R: -Z/+Z
  1/Q: J1-/J1+     2/E: J2-/J2+     3/T: J3-/J3+
  4/Y: J4-/J4+     5/U: J5-/J5+
  H: slow joint-space HOME     SPACE: stop/hold     ESC: torque off
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SAFE_ROOT = REPO_ROOT / "7_FK_safe_zone"
HARDWARE_ROOT = REPO_ROOT / "9_robot_teaching_data_collection"
OFFSET_PATH = REPO_ROOT / "config" / "jdcobot200" / "offsets.txt"
LIMIT_PATH = REPO_ROOT / "1_jdcobot200_hardware_control" / "joint_limits.txt"
sys.path[:0] = [str(ROOT), str(SAFE_ROOT), str(HARDWARE_ROOT)]

from jdcobot200_safe_mink_ik import (  # noqa: E402
    CENTER_TICK,
    DEFAULT_JOINT_SIGNS,
    MOTOR_IDS,
    load_int_map,
    parse_signs,
    radians_to_ticks,
    read_positions,
    ticks_to_radians,
)
from motor_drive import MiniFeetechDriver  # noqa: E402
from safety_model import ARM_JOINT_NAMES, JDCobotSafetyModel  # noqa: E402


CARTESIAN_KEYS = {
    "a": np.array([-1.0, 0.0, 0.0]), "d": np.array([1.0, 0.0, 0.0]),
    "s": np.array([0.0, -1.0, 0.0]), "w": np.array([0.0, 1.0, 0.0]),
    "f": np.array([0.0, 0.0, -1.0]), "r": np.array([0.0, 0.0, 1.0]),
}
JOINT_KEYS = {
    "1": (0, -1.0), "q": (0, 1.0),
    "2": (1, -1.0), "e": (1, 1.0),
    "3": (2, -1.0), "t": (2, 1.0),
    "4": (3, -1.0), "y": (3, 1.0),
    "5": (4, -1.0), "u": (4, 1.0),
}
# Wrist-pitch manual +/- is reversed on the physical robot. This applies only
# to direct joint jogging; Cartesian IK still uses the calibrated joint signs.
MANUAL_JOINT_DIRECTIONS = np.array([1.0, 1.0, 1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JDCobot200 저속 수동 IK UI")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument(
        "--cartesian-step-mm", type=float, choices=(1.0, 5.0, 10.0), default=1.0,
        help="UI 시작 시 선택할 Cartesian 이동 단위(mm: 1, 5, 10; 기본 1)",
    )
    parser.add_argument(
        "--joint-step-deg", type=float, default=1.0,
        help="관절 수동 조작 1회 변화량(deg, 기본 1)",
    )
    parser.add_argument(
        "--max-joint-speed-deg-s", type=float, default=5.0,
        help="명령 관절 최대 속도(deg/s, 기본 5)",
    )
    parser.add_argument(
        "--encoder-period", type=float, default=0.35,
        help="백그라운드 엔코더 갱신 주기(초, 기본 0.35)",
    )
    parser.add_argument("--joint-signs", default=DEFAULT_JOINT_SIGNS)
    return parser.parse_args()


class ManualIKUI:
    UPDATE_MS = 40

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.safety = JDCobotSafetyModel()
        self.signs = parse_signs(args.joint_signs)
        offsets = load_int_map(OFFSET_PATH)
        limits = load_int_map(LIMIT_PATH, pair=True)
        self.centers = np.array([
            CENTER_TICK + int(offsets[int(i)]) for i in MOTOR_IDS
        ])
        self.lower_ticks = np.array([
            limits[int(i)][0] + int(offsets[int(i)])  # type: ignore[index]
            for i in MOTOR_IDS
        ], dtype=int)
        self.upper_ticks = np.array([
            limits[int(i)][1] + int(offsets[int(i)])  # type: ignore[index]
            for i in MOTOR_IDS
        ], dtype=int)
        self.driver = MiniFeetechDriver(args.port, args.baudrate)
        ticks = read_positions(self.driver)
        self.measured_q = ticks_to_radians(ticks, self.centers, self.signs)
        self.command_q = self.measured_q.copy()
        self.goal_q = self.measured_q.copy()
        self.last_sent_ticks = ticks.copy()
        self.armed = False
        self.closed = False
        self.last_update = time.perf_counter()
        self.feedback_lock = threading.Lock()
        self.feedback_stop = threading.Event()
        self.feedback_ok_count = 0
        self.feedback_error_count = 0
        self.feedback_consecutive_errors = 0
        self.feedback_last_ok = time.perf_counter()
        self.feedback_last_error = ""
        self.motion_active = False
        self.motion_started = 0.0
        self.motion_timeout = 0.0
        self.motion_stable_reads = 0
        self.motion_last_feedback_count = 0
        self.motion_label = ""
        self.motion_buttons: list[ttk.Button] = []
        self.cartesian_step = tk.DoubleVar(value=args.cartesian_step_mm)
        self.status = tk.StringVar(value="DISARMED — 먼저 ARM 버튼을 누르세요.")
        self.communication = tk.StringVar(value="COMM: starting feedback thread")
        self.pose = tk.StringVar()
        self.goal = tk.StringVar()
        self._build()
        self._set_motion_controls(False)
        self._refresh_text()
        root.bind("<KeyPress>", self._on_key)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.feedback_thread = threading.Thread(
            target=self._feedback_loop, name="jdcobot-feedback", daemon=True
        )
        self.feedback_thread.start()
        root.after(self.UPDATE_MS, self._update)

    def _build(self) -> None:
        self.root.title("JDCobot200 Manual IK — SLOW DIAGNOSTIC CONTROL")
        self.root.geometry("900x650")
        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="수동 IK 진단 제어 (충돌/바닥/특이점 검사 OFF)",
            foreground="#b00020", font=("Sans", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="주변을 비우고 한 손을 ESC 키 가까이에 두십시오. "
                 "하드웨어 관절 한계와 저속 제한은 항상 적용됩니다.",
        ).pack(anchor="w", pady=(4, 12))

        controls = ttk.Frame(frame)
        controls.pack(fill="x")
        ttk.Button(controls, text="ARM / Torque ON", command=self.arm).pack(side="left")
        ttk.Button(controls, text="STOP / Hold (Space)", command=self.stop).pack(side="left", padx=8)
        ttk.Button(controls, text="E-STOP / Torque OFF (Esc)", command=self.emergency_stop).pack(side="left")
        home_button = ttk.Button(controls, text="Slow HOME (H)", command=self.go_home)
        home_button.pack(side="left", padx=8)
        self.motion_buttons.append(home_button)

        ttk.Separator(frame).pack(fill="x", pady=14)
        ttk.Label(frame, textvariable=self.status, font=("Sans", 12, "bold")).pack(anchor="w")
        ttk.Label(frame, textvariable=self.communication).pack(anchor="w", pady=(4, 0))
        ttk.Label(frame, textvariable=self.pose, font=("Monospace", 11)).pack(anchor="w", pady=(12, 4))
        ttk.Label(frame, textvariable=self.goal, font=("Monospace", 11)).pack(anchor="w")

        xyz = ttk.LabelFrame(frame, text="Cartesian IK jog — base/world 좌표", padding=10)
        xyz.pack(fill="x", pady=14)
        step_box = ttk.Frame(xyz)
        step_box.pack(fill="x", pady=(0, 8))
        ttk.Label(step_box, text="이동 단위:").pack(side="left")
        ttk.Radiobutton(
            step_box, text="1 mm", variable=self.cartesian_step, value=1.0,
            command=self._step_changed,
        ).pack(side="left", padx=(8, 2))
        ttk.Radiobutton(
            step_box, text="5 mm", variable=self.cartesian_step, value=5.0,
            command=self._step_changed,
        ).pack(side="left", padx=2)
        ttk.Radiobutton(
            step_box, text="10 mm", variable=self.cartesian_step, value=10.0,
            command=self._step_changed,
        ).pack(side="left", padx=2)
        xyz_buttons = ttk.Frame(xyz)
        xyz_buttons.pack(fill="x")
        for text, key in (("−X (A)", "a"), ("+X (D)", "d"),
                          ("−Y (S)", "s"), ("+Y (W)", "w"),
                          ("−Z (F)", "f"), ("+Z (R)", "r")):
            button = ttk.Button(
                xyz_buttons, text=text,
                command=lambda k=key: self.cartesian_jog(k),
            )
            button.pack(side="left", padx=4)
            self.motion_buttons.append(button)

        joints = ttk.LabelFrame(frame, text="Joint jog — HOME 특이점 탈출용", padding=10)
        joints.pack(fill="x")
        for index, name in enumerate(ARM_JOINT_NAMES):
            box = ttk.Frame(joints)
            box.pack(side="left", padx=7)
            ttk.Label(box, text=name).pack()
            minus_key = str(index + 1)
            plus_key = ("q", "e", "t", "y", "u")[index]
            minus_button = ttk.Button(
                box, text=f"− ({minus_key.upper()})",
                command=lambda k=minus_key: self.joint_jog(k),
            )
            minus_button.pack(side="left")
            plus_button = ttk.Button(
                box, text=f"+ ({plus_key.upper()})",
                command=lambda k=plus_key: self.joint_jog(k),
            )
            plus_button.pack(side="left")
            self.motion_buttons.extend((minus_button, plus_button))

        ttk.Label(
            frame,
            text=(f"설정: Cartesian 1/5/10 mm 선택, "
                  f"Joint {self.args.joint_step_deg:g}°/click, "
                  f"Speed ≤ {self.args.max_joint_speed_deg_s:g}°/s"),
        ).pack(anchor="w", pady=14)

    def _step_changed(self) -> None:
        self.status.set(f"Cartesian 이동 단위: {self.cartesian_step.get():g} mm")

    def arm(self) -> None:
        if self.armed:
            return
        if not messagebox.askyesno(
            "Torque ON 확인",
            "로봇 주변에 사람과 장애물이 없습니까?\n저속 수동 제어를 시작할까요?",
        ):
            return
        ticks = read_positions(self.driver)
        self.measured_q = ticks_to_radians(ticks, self.centers, self.signs)
        self.command_q = self.measured_q.copy()
        self.goal_q = self.measured_q.copy()
        self.last_sent_ticks = ticks.copy()
        for motor_id, tick in zip(MOTOR_IDS, ticks):
            self.driver.set_position(int(motor_id), int(tick))
            if not self.driver.set_torque_verified(int(motor_id), True):
                self.emergency_stop()
                messagebox.showerror("오류", f"모터 ID {motor_id} Torque ON 실패")
                return
            time.sleep(0.04)
        self.armed = True
        self._set_motion_controls(True)
        self.status.set("ARMED — 저속 수동 제어 가능")

    def stop(self) -> None:
        self.motion_active = False
        self._set_motion_controls(True)
        self.goal_q = self.measured_q.copy()
        self.command_q = self.measured_q.copy()
        if self.armed:
            self._send(self.command_q)
            self.status.set("STOP/HOLD — 현재 실측 위치 유지")

    def emergency_stop(self) -> None:
        self.motion_active = False
        self._set_motion_controls(False)
        self.armed = False
        for motor_id in MOTOR_IDS:
            try:
                self.driver.set_torque_verified(int(motor_id), False)
            except Exception:
                pass
        self.status.set("E-STOP — Torque OFF")

    def go_home(self) -> None:
        if not self._require_ready_for_motion():
            return
        start_goal = self.goal_q.copy()
        self.goal_q = np.zeros(5, dtype=float)
        self._begin_motion("SLOW HOME", start_goal)

    def cartesian_jog(self, key: str) -> None:
        if not self._require_ready_for_motion():
            return
        direction = CARTESIAN_KEYS[key]
        start = self.goal_q.copy()
        current_tcp = self.safety.evaluate(start, 0.0, 0.0).tcp_position
        target_tcp = current_tcp + direction * (self.cartesian_step.get() / 1000.0)
        try:
            self.status.set(
                f"IK 계산 중 — {key.upper()} {self.cartesian_step.get():g} mm"
            )
            self.root.update_idletasks()
            solution, error, candidate_count = self._solve_manual_cartesian(
                start, target_tcp
            )
            if error > 0.001:
                raise RuntimeError(
                    f"도달 가능한 해 없음: 최소 오차 {error * 1000:.2f} mm"
                )
            self.goal_q = np.clip(solution, self.safety.lower, self.safety.upper)
            self._begin_motion(
                f"IK {key.upper()} {self.cartesian_step.get():g} mm "
                f"({candidate_count} candidates)", start
            )
        except Exception as exc:
            self.status.set(f"IK 실패: {exc}")

    def _solve_manual_cartesian(
        self, start_q: np.ndarray, target_tcp: np.ndarray
    ) -> tuple[np.ndarray, float, int]:
        """Multi-start position IK for manual 1/5/10 mm jogging."""
        rng = np.random.default_rng(20260907)
        seeds = [start_q[:4].copy()]
        # Local seeds escape the zero-gradient direction at the vertical HOME
        # singularity without forcing a fixed elbow branch.
        for scale_deg in (2.0, 5.0, 15.0, 35.0):
            for _ in range(4):
                seed = start_q[:4] + np.deg2rad(
                    rng.normal(0.0, scale_deg, size=4)
                )
                seeds.append(np.clip(seed, self.safety.lower[:4], self.safety.upper[:4]))

        results = []

        def residual(q4: np.ndarray) -> np.ndarray:
            q = np.r_[q4, start_q[4]]
            position = self.safety.evaluate(q, 0.0, 0.0).tcp_position
            return (position - target_tcp) * 1000.0

        for seed in seeds:
            solved = least_squares(
                residual, seed,
                bounds=(self.safety.lower[:4], self.safety.upper[:4]),
                max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
            q = np.r_[solved.x, start_q[4]]
            error = float(np.linalg.norm(residual(solved.x))) / 1000.0
            distance = float(np.linalg.norm(q - start_q))
            results.append((error, distance, q))

        accurate = [item for item in results if item[0] <= 0.001]
        if accurate:
            error, _, solution = min(accurate, key=lambda item: item[1])
            return solution, error, len(accurate)
        error, _, solution = min(results, key=lambda item: item[0])
        return solution, error, 0

    def joint_jog(self, key: str) -> None:
        if not self._require_ready_for_motion():
            return
        index, sign = JOINT_KEYS[key]
        start_goal = self.goal_q.copy()
        corrected_sign = sign * MANUAL_JOINT_DIRECTIONS[index]
        self.goal_q[index] += np.deg2rad(
            corrected_sign * self.args.joint_step_deg
        )
        self.goal_q = np.clip(self.goal_q, self.safety.lower, self.safety.upper)
        self._begin_motion(
            f"Joint jog: {ARM_JOINT_NAMES[index]} {'+' if sign > 0 else '-'}",
            start_goal,
        )

    def _require_armed(self) -> bool:
        if not self.armed:
            self.status.set("DISARMED — ARM 버튼을 먼저 누르세요.")
            return False
        return True

    def _require_ready_for_motion(self) -> bool:
        if not self._require_armed():
            return False
        if self.motion_active:
            self.status.set(f"{self.motion_label} 동작 완료를 기다리는 중")
            return False
        return True

    def _set_motion_controls(self, enabled: bool) -> None:
        state = "normal" if enabled and self.armed else "disabled"
        for button in self.motion_buttons:
            button.configure(state=state)

    def _begin_motion(self, label: str, start_goal: np.ndarray) -> None:
        change_deg = float(np.max(np.abs(np.rad2deg(self.goal_q - start_goal))))
        expected = change_deg / self.args.max_joint_speed_deg_s
        self.motion_active = True
        self.motion_started = time.perf_counter()
        self.motion_timeout = max(5.0, expected + 5.0)
        self.motion_stable_reads = 0
        self.motion_last_feedback_count = self.feedback_ok_count
        self.motion_label = label
        self._set_motion_controls(False)
        self.status.set(f"MOVING — {label} (버튼 잠금)")

    def _check_motion_complete(self, now: float) -> None:
        if not self.motion_active:
            return
        with self.feedback_lock:
            feedback_count = self.feedback_ok_count
            measured_q = self.measured_q.copy()
        if feedback_count != self.motion_last_feedback_count:
            self.motion_last_feedback_count = feedback_count
            command_done = np.max(np.abs(self.goal_q - self.command_q)) < np.deg2rad(0.15)
            measured_done = np.max(np.abs(self.goal_q - measured_q)) < np.deg2rad(1.5)
            self.motion_stable_reads = (
                self.motion_stable_reads + 1
                if command_done and measured_done else 0
            )
        if self.motion_stable_reads >= 2:
            label = self.motion_label
            self.motion_active = False
            self._set_motion_controls(True)
            self.status.set(f"DONE — {label} 완료 (다음 입력 가능)")
        elif now - self.motion_started > self.motion_timeout:
            error_deg = np.max(np.abs(np.rad2deg(self.goal_q - measured_q)))
            label = self.motion_label
            self.motion_active = False
            self._set_motion_controls(True)
            self.status.set(
                f"TIMEOUT — {label}, 목표 오차 {error_deg:.2f}° (버튼 잠금 해제)"
            )

    def _on_key(self, event) -> None:
        key = event.keysym.lower()
        if key == "escape":
            self.emergency_stop()
        elif key == "space":
            self.stop()
        elif key == "h":
            self.go_home()
        elif key in CARTESIAN_KEYS:
            self.cartesian_jog(key)
        elif key in JOINT_KEYS:
            self.joint_jog(key)

    def _send(self, q: np.ndarray) -> None:
        ticks = radians_to_ticks(q, self.centers, self.signs)
        ticks = np.clip(ticks, self.lower_ticks, self.upper_ticks).astype(int)
        for index, (motor_id, tick) in enumerate(zip(MOTOR_IDS, ticks)):
            if tick != self.last_sent_ticks[index]:
                self.driver.set_position(int(motor_id), int(tick))
                self.last_sent_ticks[index] = tick

    def _feedback_loop(self) -> None:
        """Read each encoder once without ever blocking Tk's event loop."""
        while not self.feedback_stop.is_set():
            values = []
            try:
                for motor_id in MOTOR_IDS:
                    value = self.driver.get_position(int(motor_id))
                    if value is None:
                        raise RuntimeError(f"ID {motor_id} 응답 없음")
                    values.append(value)
                measured = ticks_to_radians(
                    np.asarray(values, dtype=int), self.centers, self.signs
                )
                with self.feedback_lock:
                    self.measured_q = measured
                    self.feedback_ok_count += 1
                    self.feedback_consecutive_errors = 0
                    self.feedback_last_ok = time.perf_counter()
                    self.feedback_last_error = ""
            except Exception as exc:
                with self.feedback_lock:
                    self.feedback_error_count += 1
                    self.feedback_consecutive_errors += 1
                    self.feedback_last_error = str(exc)
            self.feedback_stop.wait(self.args.encoder_period)

    def _update(self) -> None:
        if self.closed:
            return
        now = time.perf_counter()
        dt = min(0.1, max(0.001, now - self.last_update))
        self.last_update = now
        if self.armed:
            max_step = np.deg2rad(self.args.max_joint_speed_deg_s) * dt
            delta = np.clip(self.goal_q - self.command_q, -max_step, max_step)
            self.command_q = np.clip(
                self.command_q + delta, self.safety.lower, self.safety.upper
            )
            self._send(self.command_q)
            self._check_motion_complete(now)
        self._refresh_text()
        self.root.after(self.UPDATE_MS, self._update)

    def _refresh_text(self) -> None:
        with self.feedback_lock:
            measured_q = self.measured_q.copy()
            ok_count = self.feedback_ok_count
            error_count = self.feedback_error_count
            consecutive = self.feedback_consecutive_errors
            age = time.perf_counter() - self.feedback_last_ok
            last_error = self.feedback_last_error
        tcp = self.safety.evaluate(measured_q, 0.0, 0.0).tcp_position * 1000.0
        goal_tcp = self.safety.evaluate(self.goal_q, 0.0, 0.0).tcp_position * 1000.0
        self.pose.set(
            f"MEASURED q(deg): {np.round(np.rad2deg(measured_q), 2)}\n"
            f"MEASURED TCP(mm): {np.round(tcp, 2)}"
        )
        self.goal.set(
            f"GOAL     q(deg): {np.round(np.rad2deg(self.goal_q), 2)}\n"
            f"GOAL     TCP(mm): {np.round(goal_tcp, 2)}"
        )
        state = "OK" if consecutive == 0 and age < 2.0 else "DELAY/ERROR"
        detail = f" | last={last_error}" if last_error else ""
        self.communication.set(
            f"COMM: {state} | ok={ok_count} errors={error_count} "
            f"consecutive={consecutive} last_ok={age:.1f}s ago{detail}"
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.feedback_stop.set()
        self.feedback_thread.join(timeout=1.0)
        self.emergency_stop()
        self.driver.close()
        self.root.destroy()


def main() -> int:
    args = parse_args()
    if min(args.cartesian_step_mm, args.joint_step_deg,
           args.max_joint_speed_deg_s, args.encoder_period) <= 0:
        raise ValueError("step과 speed는 양수여야 합니다.")
    if args.cartesian_step_mm > 10.0:
        raise ValueError("수동 Cartesian step은 10 mm 이하로 설정하세요.")
    if args.joint_step_deg > 5.0 or args.max_joint_speed_deg_s > 15.0:
        raise ValueError("수동 joint step은 5°, 속도는 15°/s 이하로 설정하세요.")
    root = tk.Tk()
    ManualIKUI(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
