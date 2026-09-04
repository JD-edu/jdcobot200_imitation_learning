import time
import math
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from pathlib import Path
from serial.tools import list_ports
from motor_control import MiniFeetechDriver

CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "config" / "jdcobot200" / "offsets.txt"

class JdCobotTeachingUI:
    def __init__(self, window):
        self.window = window
        self.window.title("jdcobot200 수동 티칭 및 플레이백 시스템 (칼리브레이션 적용)")
        self.window.geometry("900x720")
        
        # --- 하드웨어 설정 ---
        self.PORT = ""
        self.BAUDRATE = 1000000
        self.MOTOR_IDS = [1, 2, 3, 4, 5, 6]
        self.DEG_TO_TICK = 4096.0 / 360.0
        self.TICK_TO_DEG = 360.0 / 4096.0
        self.THEORETICAL_CENTER = 2048 # STS3215의 이론상 물리적 중심점 [cite: 60]
        
        # --- 칼리브레이션 오프셋 데이터 공간 ---
        self.offsets = {m_id: 0 for m_id in self.MOTOR_IDS}
        self.load_offsets_from_file() # 프로그램 시작 시 offsets.txt 읽기 [cite: 5]
        
        self.driver = None
        self.is_connected = False

        # --- 제어 변수 ---
        self.center_positions = {}    # 칼리브레이션 원점 (2048 + 오프셋) [cite: 6]
        self.current_positions = {}   # 로봇의 현재 타겟/실제 물리 위치 (Tick)
        self.saved_sequence = []      # 저장된 티칭 포인트 리스트 (각도 리스트 형태)
        self.is_moving = False        # 자동 모션 작동 중 플래그
        self.stop_requested = False   # 동작 취소 및 비상 정지 플래그
        self.torque_state = True      # 현재 토크 상태
        self.playback_running = False
        self.continuous_playback = False
        self.feedback_paused = threading.Event()
        self.command_lock = threading.RLock()
        
        # 초기화 동작 및 안전 가감속 프로파일 세팅 [cite: 36, 46]
        for m_id in self.MOTOR_IDS:
            # 소프트웨어 보정 방식: 이론상 중심(2048)에 파일에서 읽은 오프셋 적용 [cite: 6]
            software_home = self.THEORETICAL_CENTER + self.offsets[m_id]
            self.center_positions[m_id] = software_home
            self.current_positions[m_id] = software_home
            
        # --- UI 레이아웃 생성 ---
        self.create_widgets()
        self.refresh_serial_ports(show_message=False)
        
        # 실시간 각도 피드백을 위한 데몬 스레드 시작
        self.feedback_running = True
        self.feedback_thread = threading.Thread(target=self.update_angle_feedback_loop, daemon=True)
        self.feedback_thread.start()

    def load_offsets_from_file(self, file_path=CALIBRATION_FILE):
        """ 텍스트 파일에서 모터 ID와 오프셋 매핑 데이터를 로드합니다. """ 
        if not os.path.exists(file_path):
            print(f"⚠️ [{file_path}] 파일이 없어 기본 오프셋(0)을 사용합니다.")
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    if "=" in line:
                        motor_id, offset_val = line.split("=")
                        m_id = int(motor_id)
                        if m_id in self.MOTOR_IDS:
                            self.offsets[m_id] = int(offset_val)
            print(f"▶ [{file_path}] 오프셋 데이터 로드 완료: {self.offsets}")
        except Exception as e:
            print(f"❌ 오프셋 파일 로드 중 오류 발생: {e}")

    def refresh_serial_ports(self, show_message=True):
        """Scan Windows COM ports and Linux/macOS serial devices."""
        previous_device = self.PORT
        ports = sorted(list_ports.comports(), key=lambda item: item.device)
        self.port_devices = {
            f"{port.device} — {port.description}": port.device for port in ports
        }
        labels = list(self.port_devices)
        self.port_combo["values"] = labels
        selected = next(
            (label for label, device in self.port_devices.items()
             if device == previous_device),
            labels[0] if labels else "",
        )
        self.port_combo.set(selected)
        if not labels and show_message:
            messagebox.showwarning(
                "포트 없음",
                "사용 가능한 시리얼 포트를 찾지 못했습니다.\n"
                "USB 연결과 장치 드라이버를 확인한 후 새로고침하세요.",
            )
        print("[시리얼 포트]", [port.device for port in ports])

    def connect_selected_port(self):
        if self.is_connected:
            if self.is_moving or self.playback_running:
                messagebox.showwarning("연결 해제 불가", "로봇 동작을 먼저 정지하세요.")
                return
            self.feedback_paused.set()
            try:
                self.driver.close()
            finally:
                self.driver = None
                self.is_connected = False
                self.lbl_connection.config(text="연결 안 됨", fg="#c0392b")
                self.btn_connect.config(text="🔌 연결", bg="#2980b9")
                self.feedback_paused.clear()
            print(f"[연결 해제] {self.PORT}")
            return

        label = self.port_combo.get()
        port = self.port_devices.get(label)
        if not port:
            messagebox.showwarning("연결 불가", "먼저 시리얼 포트를 선택하세요.")
            return

        self.feedback_paused.set()
        candidate = None
        try:
            candidate = MiniFeetechDriver(port, self.BAUDRATE)
            # Read every servo first. A partial bus connection is not accepted.
            positions = {}
            for motor_id in self.MOTOR_IDS:
                position = candidate.get_position_filtered(motor_id, samples=3)
                if position is None:
                    raise RuntimeError(f"서보 ID {motor_id}의 응답이 없습니다.")
                positions[motor_id] = position

            # Prevent a jump: write each measured pose before enabling torque.
            for motor_id in self.MOTOR_IDS:
                candidate.set_position(motor_id, positions[motor_id])
                if not candidate.set_torque_verified(motor_id, True):
                    raise RuntimeError(f"서보 ID {motor_id}의 Torque ON 확인 실패")
                if hasattr(candidate, "set_acceleration"):
                    candidate.set_acceleration(motor_id, 40)
                if hasattr(candidate, "set_speed"):
                    candidate.set_speed(motor_id, 1000)
                time.sleep(0.05)
        except Exception as exc:
            if candidate is not None:
                candidate.close()
            messagebox.showerror("연결 실패", f"{port}에 연결하지 못했습니다.\n{exc}")
            print(f"[연결 실패] {port}: {exc}")
            return
        finally:
            self.feedback_paused.clear()

        self.driver = candidate
        self.PORT = port
        self.current_positions.update(positions)
        self.is_connected = True
        self.torque_state = True
        self.lbl_connection.config(text=f"연결됨: {port}", fg="#27ae60")
        self.btn_connect.config(text="연결 해제", bg="#7f8c8d")
        print(f"[연결 성공] {port}, baudrate={self.BAUDRATE}")

    def create_widgets(self):
        connection_frame = tk.LabelFrame(self.window, text=" 시리얼 포트 연결 ", padx=8, pady=6)
        connection_frame.pack(fill=tk.X, padx=10, pady=(8, 0))
        self.port_combo = ttk.Combobox(connection_frame, state="readonly", width=22)
        self.port_combo.pack(side=tk.LEFT, padx=4)
        tk.Button(
            connection_frame, text="🔄 포트 새로고침", command=self.refresh_serial_ports
        ).pack(side=tk.LEFT, padx=4)
        self.btn_connect = tk.Button(
            connection_frame, text="🔌 연결", bg="#2980b9", fg="white",
            command=self.connect_selected_port,
        )
        self.btn_connect.pack(side=tk.LEFT, padx=4)
        self.lbl_connection = tk.Label(connection_frame, text="연결 안 됨", fg="#c0392b")
        self.lbl_connection.pack(side=tk.LEFT, padx=10)

        # 상단 제어 바 (홈 / 비상중지)
        top_frame = tk.Frame(self.window, pady=10)
        top_frame.pack(fill=tk.X)
        
        self.btn_home = tk.Button(top_frame, text="🏠 홈 위치 이동", bg="#2ecc71", fg="white", font=('Arial', 11, 'bold'), width=15, command=self.go_home)
        self.btn_home.pack(side=tk.LEFT, padx=20)
        
        self.btn_stop = tk.Button(top_frame, text="🚨 모든동작 취소 / 중지", bg="#e74c3c", fg="white", font=('Arial', 11, 'bold'), width=22, command=self.emergency_stop)
        self.btn_stop.pack(side=tk.RIGHT, padx=20)

        # 메인 프레임 분할 (좌측: 모니터링 및 미세조정, 우측: 티칭 제어창)
        main_frame = tk.Frame(self.window, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.LabelFrame(main_frame, text=" 1~2. 관절별 현재 각도 표시 및 미세 조절 (칼리브레이션 반영) ", padx=15, pady=15)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_frame = tk.LabelFrame(main_frame, text=" 3~5. 수동 티칭 및 플레이백 ", padx=15, pady=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)

        # --- 좌측 프레임 컴포넌트 (실시간 각도 정보창 및 ±3도 버튼) ---
        self.lbl_angles = {}
        for m_id in self.MOTOR_IDS:
            row_frame = tk.Frame(left_frame, pady=8)
            row_frame.pack(fill=tk.X)
            
            lbl_name = tk.Label(row_frame, text=f"관절 {m_id}:", font=('Arial', 10, 'bold'), width=7, anchor='w')
            lbl_name.pack(side=tk.LEFT)
            
            # 3도 미세 조정 버튼
            btn_dec = tk.Button(row_frame, text="-3°", width=5, command=lambda m=m_id: self.jog_joint(m, -3))
            btn_dec.pack(side=tk.LEFT, padx=3)
            
            btn_inc = tk.Button(row_frame, text="+3°", width=5, command=lambda m=m_id: self.jog_joint(m, 3))
            btn_inc.pack(side=tk.LEFT, padx=3)
            
            # 실시간 피드백 디스플레이 라벨
            lbl_ang = tk.Label(row_frame, text="0.0°", font=('Courier', 11, 'bold'), fg="#2c3e50", width=12, anchor='e')
            lbl_ang.pack(side=tk.RIGHT, padx=10)
            self.lbl_angles[m_id] = lbl_ang

        # --- 우측 프레임 컴포넌트 (토크 제어, 저장, 재생) ---
        torque_frame = tk.Frame(right_frame, pady=5)
        torque_frame.pack(fill=tk.X)
        
        self.btn_torque_off = tk.Button(torque_frame, text="🔓 수동 조작 (Torque OFF)", bg="#e67e22", fg="white", font=('Arial', 10, 'bold'), width=18, command=self.torque_off)
        self.btn_torque_off.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.btn_torque_on = tk.Button(torque_frame, text="🔒 위치 고정 (Torque ON)", bg="#34495e", fg="white", font=('Arial', 10, 'bold'), width=18, command=self.torque_on)
        self.btn_torque_on.pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.X)

        # 저장 리스트박스 및 관리 버튼
        self.listbox = tk.Listbox(right_frame, width=38, height=14, font=('Courier', 9), bg="#f8f9fa")
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=10)
        
        btn_save = tk.Button(right_frame, text="💾 현재 위치 티칭 포인트 저장", bg="#9b59b6", fg="white", font=('Arial', 11, 'bold'), pady=5, command=self.save_teaching_point)
        btn_save.pack(fill=tk.X, pady=3)
        
        btn_clear = tk.Button(right_frame, text="🗑️ 티칭 리스트 전체 삭제", bg="#7f8c8d", fg="white", command=self.clear_sequence)
        btn_clear.pack(fill=tk.X, pady=2)
        
        file_frame = tk.Frame(right_frame)
        file_frame.pack(fill=tk.X, pady=3)
        tk.Button(
            file_frame, text="💾 지금 시퀀스 저장하기", bg="#2980b9", fg="white",
            command=self.save_sequence_to_file,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        tk.Button(
            file_frame, text="📂 저장된 시퀀스 불러오기", bg="#16a085", fg="white",
            command=self.load_sequence_from_file,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        self.btn_play = tk.Button(
            right_frame, text="▶ 저장된 시퀀스 1회 재생", bg="#f1c40f", fg="black",
            font=('Arial', 11, 'bold'), pady=6, command=self.start_playback_thread,
        )
        self.btn_play.pack(fill=tk.X, pady=(10, 3))

        continuous_frame = tk.Frame(right_frame)
        continuous_frame.pack(fill=tk.X, pady=3)
        tk.Button(
            continuous_frame, text="🔁 연속 재생", bg="#27ae60", fg="white",
            font=('Arial', 10, 'bold'), command=self.start_continuous_playback,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        tk.Button(
            continuous_frame, text="⏹ 연속 재생 정지", bg="#c0392b", fg="white",
            font=('Arial', 10, 'bold'), command=self.stop_continuous_playback,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

    # --- 실시간 수동 위치 데이터 수집 루프 ---
    def update_angle_feedback_loop(self):
        """ 수동 조작 상태일 때, 오프셋 원점을 기준으로 각도를 연산하여 화면에 업데이트합니다. """
        while self.feedback_running:
            if self.is_connected and not self.feedback_paused.is_set():
                with self.command_lock:
                    for m_id in self.MOTOR_IDS:
                        if not self.is_moving:
                            pos = self.driver.get_position_filtered(m_id, samples=3)
                            if pos is not None:
                                self.current_positions[m_id] = pos
            
            # 오프셋 원점(center_positions)을 기준으로 상대적 각도 계산 [cite: 6]
            for m_id in self.MOTOR_IDS:
                tick_offset = self.current_positions[m_id] - self.center_positions[m_id] 
                deg = tick_offset * self.TICK_TO_DEG
                try:
                    self.lbl_angles[m_id].config(text=f"{deg:+.1f}°")
                except:
                    pass
            time.sleep(0.1)

    # --- 토크 제어 (수동 조작 스위칭) ---
    def torque_off(self, force=False):
        if self.is_moving and not force:
            return
        print("[티칭] 프리 무브 활성화: 토크 OFF")
        failed = []
        if self.is_connected:
            self.feedback_paused.set()
            try:
                with self.command_lock:
                    for m_id in self.MOTOR_IDS:
                        if not self.driver.set_torque_verified(m_id, False):
                            failed.append(m_id)
                        time.sleep(0.05)
            finally:
                self.feedback_paused.clear()
        self.torque_state = False
        if failed:
            self.torque_state = True
            messagebox.showerror(
                "Torque OFF 실패",
                f"토크 해제를 확인하지 못한 서보: {failed}\n"
                "로봇을 강제로 움직이지 말고 연결 상태를 확인하세요.",
            )
            return
        self.btn_torque_off.config(relief=tk.SUNKEN, bg="#d35400")
        self.btn_torque_on.config(relief=tk.RAISED, bg="#34495e")

    def torque_on(self):
        print("[티칭] 위치 고정 활성화: 토크 ON")
        failed = []
        if self.is_connected:
            self.feedback_paused.set()
            try:
                with self.command_lock:
                    for m_id in self.MOTOR_IDS:
                        pos = self.driver.get_position_filtered(m_id, samples=5)
                        if pos is not None:
                            self.driver.set_position(m_id, pos)
                            self.current_positions[m_id] = pos
                        if not self.driver.set_torque_verified(m_id, True):
                            failed.append(m_id)
                        time.sleep(0.05)
            finally:
                self.feedback_paused.clear()
        self.torque_state = not failed
        if failed:
            messagebox.showerror(
                "Torque ON 실패",
                f"토크 활성화를 확인하지 못한 서보: {failed}",
            )
            return
        self.btn_torque_off.config(relief=tk.RAISED, bg="#e67e22")
        self.btn_torque_on.config(relief=tk.SUNKEN, bg="#2c3e50")

    # --- 티칭 데이터 관리 기능 ---
    def save_teaching_point(self):
        """ 칼리브레이션 원점 기준의 정밀 각도를 계산하여 티칭 포인트로 저장합니다. """
        current_angles = []
        for m_id in self.MOTOR_IDS:
            tick_offset = self.current_positions[m_id] - self.center_positions[m_id] 
            deg = round(tick_offset * self.TICK_TO_DEG, 1)
            current_angles.append(deg)
            
        self.saved_sequence.append(current_angles)
        
        idx = len(self.saved_sequence)
        ang_str = ", ".join([f"{a:+.1f}°" for a in current_angles])
        self.listbox.insert(tk.END, f"포인트 {idx:02d} ➡️ [{ang_str}]")
        print(f"[티칭 저장] P{idx:02d} 완료")

    def clear_sequence(self):
        if self.playback_running:
            messagebox.showwarning("삭제 불가", "재생을 먼저 정지하세요.")
            return
        self.saved_sequence.clear()
        self.listbox.delete(0, tk.END)
        print("[티칭 리스트] 초기화 완료")

    def refresh_sequence_listbox(self):
        self.listbox.delete(0, tk.END)
        for index, angles in enumerate(self.saved_sequence, 1):
            angle_text = ", ".join(f"{angle:+.1f}°" for angle in angles)
            self.listbox.insert(tk.END, f"포인트 {index:02d} ➡️ [{angle_text}]")

    def save_sequence_to_file(self):
        if not self.saved_sequence:
            messagebox.showwarning("저장 불가", "저장할 티칭 시퀀스가 없습니다.")
            return
        path = filedialog.asksaveasfilename(
            title="티칭 시퀀스 저장", defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="jdcobot200_teaching_sequence.json",
        )
        if not path:
            return
        payload = {
            "format": "jdcobot200_teaching_sequence", "version": 1,
            "motor_ids": self.MOTOR_IDS, "unit": "degree",
            "sequence": self.saved_sequence,
        }
        try:
            Path(path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            messagebox.showerror("저장 실패", f"파일을 저장하지 못했습니다.\n{exc}")
            return
        print(f"[시퀀스 저장] {path}")
        messagebox.showinfo("저장 완료", f"{len(self.saved_sequence)}개 포인트를 저장했습니다.")

    def load_sequence_from_file(self):
        if self.playback_running or self.is_moving:
            messagebox.showwarning("불러오기 불가", "재생을 먼저 정지하세요.")
            return
        path = filedialog.askopenfilename(
            title="티칭 시퀀스 불러오기",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("format") != "jdcobot200_teaching_sequence":
                raise ValueError("JDCobot200 티칭 시퀀스 파일이 아닙니다.")
            if payload.get("motor_ids") != self.MOTOR_IDS:
                raise ValueError(f"motor_ids가 {self.MOTOR_IDS}와 일치하지 않습니다.")
            raw_sequence = payload.get("sequence")
            if not isinstance(raw_sequence, list) or not raw_sequence:
                raise ValueError("sequence가 비어 있거나 배열이 아닙니다.")
            loaded = []
            for index, point in enumerate(raw_sequence, 1):
                if not isinstance(point, list) or len(point) != len(self.MOTOR_IDS):
                    raise ValueError(f"포인트 {index}에 관절각 6개가 필요합니다.")
                values = [float(value) for value in point]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"포인트 {index}에 유효하지 않은 숫자가 있습니다.")
                loaded.append(values)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            messagebox.showerror("불러오기 실패", str(exc))
            return
        self.saved_sequence = loaded
        self.refresh_sequence_listbox()
        print(f"[시퀀스 불러오기] {path} ({len(loaded)}개 포인트)")
        messagebox.showinfo("불러오기 완료", f"{len(loaded)}개 포인트를 불러왔습니다.")

    # --- 모션 구동 엔진 (정현파 가감속) ---
    def drive_smooth_motion(self, target_ticks, duration=1.5):
        if not self.is_connected or self.stop_requested:
            for m_id, t_tick in target_ticks.items():
                self.current_positions[m_id] = t_tick
            return

        self.is_moving = True
        start_ticks = {m_id: self.current_positions[m_id] for m_id in self.MOTOR_IDS}
        
        control_period = 0.02
        steps = int(duration / control_period)
        
        for step in range(steps + 1):
            if self.stop_requested:
                break
                
            ratio = (1.0 - math.cos((step / steps) * math.pi)) / 2.0 
            
            with self.command_lock:
                for m_id in self.MOTOR_IDS:
                    if m_id in target_ticks:
                        current_target = start_ticks[m_id] + (target_ticks[m_id] - start_ticks[m_id]) * ratio
                        current_target = max(0, min(4095, int(current_target)))
                        self.driver.set_position(m_id, current_target)
                        self.current_positions[m_id] = current_target
                    
            time.sleep(control_period)
        self.is_moving = False

    def jog_joint(self, motor_id, deg_delta):
        if self.is_moving: return
        if not self.torque_state:
            messagebox.showwarning("동작 불가", "현재 토크가 해제 상태입니다.\n위치 고정(Torque ON)을 먼저 누르세요.")
            return
            
        self.stop_requested = False
        tick_delta = int(deg_delta * self.DEG_TO_TICK)
        target_tick = self.current_positions[motor_id] + tick_delta
        
        target_dict = {motor_id: target_tick}
        threading.Thread(target=self.drive_smooth_motion, args=(target_dict, 0.4), daemon=True).start()

    def start_playback_thread(self):
        if self.is_moving or self.playback_running:
            return
        if not self.saved_sequence:
            messagebox.showwarning("재생 불가", "저장된 티칭 포인트 데이터가 비어 있습니다.")
            return

        self.torque_on()
        if not self.torque_state:
            return
        self.stop_requested = False
        self.continuous_playback = False
        self.playback_running = True
        threading.Thread(
            target=self.run_playback_sequence, args=(False,), daemon=True
        ).start()

    def start_continuous_playback(self):
        if self.is_moving or self.playback_running:
            messagebox.showwarning("재생 중", "이미 시퀀스를 재생하고 있습니다.")
            return
        if not self.saved_sequence:
            messagebox.showwarning("재생 불가", "저장된 티칭 포인트 데이터가 비어 있습니다.")
            return
        self.torque_on()
        if not self.torque_state:
            return
        self.stop_requested = False
        self.continuous_playback = True
        self.playback_running = True
        threading.Thread(
            target=self.run_playback_sequence, args=(True,), daemon=True
        ).start()

    def stop_continuous_playback(self):
        if not self.playback_running:
            print("[연속 재생] 현재 실행 중인 재생이 없습니다.")
            return
        self.continuous_playback = False
        self.stop_requested = True
        print("[연속 재생] 정지 요청: 현재 위치에서 정지합니다.")

    def select_playback_point(self, index=None):
        self.listbox.selection_clear(0, tk.END)
        if index is not None and index < self.listbox.size():
            self.listbox.selection_set(index)
            self.listbox.see(index)

    def run_playback_sequence(self, continuous=False):
        mode = "연속" if continuous else "1회"
        print(f"[플레이백] {mode} 재생을 시작합니다.")
        repeat_count = 0
        try:
            while not self.stop_requested:
                # Playback uses a snapshot so UI-side edits cannot alter a running cycle.
                sequence_snapshot = [point[:] for point in self.saved_sequence]
                for idx, angles in enumerate(sequence_snapshot):
                    if self.stop_requested:
                        break
                    self.window.after(0, self.select_playback_point, idx)

                    target_ticks = {}
                    for i, m_id in enumerate(self.MOTOR_IDS):
                        deg = angles[i]
                        target_ticks[m_id] = (
                            self.center_positions[m_id] + int(deg * self.DEG_TO_TICK)
                        )
                    self.drive_smooth_motion(target_ticks, duration=1.6)
                    if self.stop_requested:
                        break
                    # Interruptible dwell between teaching points.
                    for _ in range(20):
                        if self.stop_requested:
                            break
                        time.sleep(0.02)

                repeat_count += 1
                if not continuous or not self.continuous_playback:
                    break
                print(f"[연속 재생] {repeat_count}회 완료")
        finally:
            self.playback_running = False
            self.continuous_playback = False
            self.is_moving = False
            self.window.after(0, self.select_playback_point, None)
            print(f"[플레이백] {mode} 재생을 종료했습니다. ({repeat_count}회 완료)")

    def go_home(self):
        """ 모든 매커니즘 정지 후 로드된 칼리브레이션 소프트웨어 원점 위치로 복귀합니다. """ 
        self.stop_requested = True
        time.sleep(0.05)
        self.torque_on()
        self.stop_requested = False
        
        print("[홈] 오프셋 칼리브레이션 홈 포지션으로 이동합니다.") 
        threading.Thread(target=self.drive_smooth_motion, args=(self.center_positions, 1.8), daemon=True).start()

    def emergency_stop(self):
        self.stop_requested = True
        print("\n🚨🚨🚨 비상 정지 명령 수신! 토크를 해제합니다. 🚨🚨🚨")
        self.torque_off(force=True)
        messagebox.showwarning("비상 중지", "자동 구동이 정지되었으며, 관절의 토크가 해제되었습니다.\n로봇을 손으로 안전하게 이송할 수 있습니다.")

if __name__ == "__main__":
    root = tk.Tk()
    app = JdCobotTeachingUI(root)
    
    def on_closing():
        app.feedback_running = False
        app.continuous_playback = False
        app.stop_requested = True
        if app.is_connected:
            app.driver.close()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
