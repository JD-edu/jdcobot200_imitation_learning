"""115번 UI의 Sync Write + minimum-jerk 개선 버전."""

import importlib.util
import math
import threading
import time
import tkinter as tk
from pathlib import Path


def _load_previous_ui():
    module_path = Path(__file__).with_name("115_robot_ui_control_cali.py")
    spec = importlib.util.spec_from_file_location("robot_ui_control_115", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"115번 UI를 불러올 수 없습니다: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.JdCobotUI


BaseJdCobotUI = _load_previous_ui()


class JdCobotUI(BaseJdCobotUI):
    """다축 명령 동시성과 궤적 시간 정확도를 개선한 UI."""

    CONTROL_PERIOD = 0.02
    # Sync Write 적용 후 사라지는 모터별 지연을 보상하고 구조 진동을 줄입니다.
    # 너무 느리면 1.1, 진동이 남으면 1.4~1.6 정도로 조정할 수 있습니다.
    MOTION_TIME_SCALE = 1.25

    def __init__(self, window):
        super().__init__(window)
        self.window.title("jdcobot200 통합 제어 시스템 (116 Sync Write)")
        self._motion_lock = threading.Lock()

    @staticmethod
    def _minimum_jerk(t):
        """시작과 끝에서 속도와 가속도가 모두 0인 5차 보간."""
        return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5

    def drive_smooth_motion(self, target_ticks, duration=1.5):
        targets = {
            motor_id: max(0, min(4095, int(position)))
            for motor_id, position in target_ticks.items()
        }
        if not targets:
            return

        if not self.is_connected or self.stop_requested:
            self.current_positions.update(targets)
            return

        # 버튼을 빠르게 연속 클릭해도 두 모션 스레드가 버스에 동시에 쓰지 않게 합니다.
        if not self._motion_lock.acquire(blocking=False):
            return

        self.is_moving = True
        try:
            duration = max(
                float(duration) * self.MOTION_TIME_SCALE,
                self.CONTROL_PERIOD,
            )
            steps = max(1, math.ceil(duration / self.CONTROL_PERIOD))
            step_period = duration / steps
            start_time = time.perf_counter()
            start_ticks = {
                motor_id: self.current_positions[motor_id] for motor_id in targets
            }

            for step in range(steps + 1):
                if self.stop_requested:
                    break

                # 누적 sleep 오차 대신 절대 시각을 기준으로 일정한 제어 주기를 유지합니다.
                deadline = start_time + step * step_period
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

                t = step / steps
                ratio = self._minimum_jerk(t)
                command = {
                    motor_id: max(
                        0,
                        min(
                            4095,
                            int(round(start + (targets[motor_id] - start) * ratio)),
                        ),
                    )
                    for motor_id, start in start_ticks.items()
                }

                # 모든 관절 목표를 하나의 브로드캐스트 패킷으로 전송합니다.
                self.driver.set_positions_sync(command)
                self.current_positions.update(command)
        finally:
            self.is_moving = False
            self._motion_lock.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = JdCobotUI(root)

    def on_closing():
        app.stop_requested = True
        if app.is_connected:
            app.driver.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
