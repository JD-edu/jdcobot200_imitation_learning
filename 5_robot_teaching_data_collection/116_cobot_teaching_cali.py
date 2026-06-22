import time
import math
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from motor_control import MiniFeetechDriver

class JdCobotTeachingUI:
    def __init__(self, window):
        self.window = window
        self.window.title("jdcobot200 수동 티칭 및 플레이백 시스템 (칼리브레이션 적용)")
        self.window.geometry("850x600")
        
        # --- 하드웨어 설정 ---
        self.PORT = "/dev/ttyACM0"
        self.BAUDRATE = 1000000
        self.MOTOR_IDS = [1, 2, 3, 4, 5, 6]
        self.DEG_TO_TICK = 4096.0 / 360.0
        self.TICK_TO_DEG = 360.0 / 4096.0
        self.THEORETICAL_CENTER = 2048 # STS3215의 이론상 물리적 중심점 [cite: 60]
        
        # --- 칼리브레이션 오프셋 데이터 공간 ---
        self.offsets = {m_id: 0 for m_id in self.MOTOR_IDS}
        self.load_offsets_from_file() # 프로그램 시작 시 offsets.txt 읽기 [cite: 5]
        
        try:
            self.driver = MiniFeetechDriver(self.PORT, self.BAUDRATE)
            self.is_connected = True
        except Exception as e:
            print(f"로봇 연결 실패: {e}")
            self.is_connected = False
            messagebox.showerror("연결 오류", f"로봇 연결 실패! 시뮬레이션 모드로 진행합니다.\n({e})")

        # --- 제어 변수 ---
        self.center_positions = {}    # 칼리브레이션 원점 (2048 + 오프셋) [cite: 6]
        self.current_positions = {}   # 로봇의 현재 타겟/실제 물리 위치 (Tick)
        self.saved_sequence = []      # 저장된 티칭 포인트 리스트 (각도 리스트 형태)
        self.is_moving = False        # 자동 모션 작동 중 플래그
        self.stop_requested = False   # 동작 취소 및 비상 정지 플래그
        self.torque_state = True      # 현재 토크 상태
        
        # 초기화 동작 및 안전 가감속 프로파일 세팅 [cite: 36, 46]
        for m_id in self.MOTOR_IDS:
            # 소프트웨어 보정 방식: 이론상 중심(2048)에 파일에서 읽은 오프셋 적용 [cite: 6]
            software_home = self.THEORETICAL_CENTER + self.offsets[m_id]
            self.center_positions[m_id] = software_home
            self.current_positions[m_id] = software_home
            
            if self.is_connected:
                # 7번 샘플링 필터링으로 초기 오차 노이즈 제거 [cite: 24]
                pos = self.driver.get_position_filtered(m_id, samples=7)
                if pos is not None:
                    self.current_positions[m_id] = pos
                self.driver.set_torque(m_id, True)
                
                # 서보 내부 하드웨어 가감속 적용 (튀는 현상 및 진동 방지) [cite: 36, 46]
                if hasattr(self.driver, 'set_acceleration'): self.driver.set_acceleration(m_id, 40)
                if hasattr(self.driver, 'set_speed'): self.driver.set_speed(m_id, 1000)

        # --- UI 레이아웃 생성 ---
        self.create_widgets()
        
        # 실시간 각도 피드백을 위한 데몬 스레드 시작
        self.feedback_running = True
        self.feedback_thread = threading.Thread(target=self.update_angle_feedback_loop, daemon=True)
        self.feedback_thread.start()

    def load_offsets_from_file(self, file_path="offsets.txt"):
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

    def create_widgets(self):
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
        
        self.btn_play = tk.Button(right_frame, text="🔁 저장된 티칭 시퀀스 재생 (Play)", bg="#f1c40f", fg="black", font=('Arial', 12, 'bold'), pady=8, command=self.start_playback_thread)
        self.btn_play.pack(fill=tk.X, pady=10)

    # --- 실시간 수동 위치 데이터 수집 루프 ---
    def update_angle_feedback_loop(self):
        """ 수동 조작 상태일 때, 오프셋 원점을 기준으로 각도를 연산하여 화면에 업데이트합니다. """
        while self.feedback_running:
            if self.is_connected:
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
    def torque_off(self):
        if self.is_moving: return
        self.torque_state = False
        print("[티칭] 프리 무브 활성화: 토크 OFF")
        if self.is_connected:
            for m_id in self.MOTOR_IDS:
                self.driver.set_torque(m_id, False)
        self.btn_torque_off.config(relief=tk.SUNKEN, bg="#d35400")
        self.btn_torque_on.config(relief=tk.RAISED, bg="#34495e")

    def torque_on(self):
        self.torque_state = True
        print("[티칭] 위치 고정 활성화: 토크 ON")
        if self.is_connected:
            for m_id in self.MOTOR_IDS:
                pos = self.driver.get_position_filtered(m_id, samples=5) 
                if pos is not None:
                    self.driver.set_position(m_id, pos)
                    self.current_positions[m_id] = pos
                self.driver.set_torque(m_id, True)
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
        self.saved_sequence.clear()
        self.listbox.delete(0, tk.END)
        print("[티칭 리스트] 초기화 완료")

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
        if self.is_moving: return
        if not self.saved_sequence:
            messagebox.showwarning("재생 불가", "저장된 티칭 포인트 데이터가 비어 있습니다.")
            return
            
        self.torque_on()
        self.stop_requested = False
        threading.Thread(target=self.run_playback_sequence, daemon=True).start()

    def run_playback_sequence(self):
        print("[플레이백] 티칭 시퀀스 자동 재생을 시작합니다.")
        for idx, angles in enumerate(self.saved_sequence):
            if self.stop_requested:
                break
                
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            
            # 저장된 각도 데이터를 오프셋 보정 원점 기준으로 역산하여 틱 배열로 생성 [cite: 6]
            target_ticks = {}
            for i, m_id in enumerate(self.MOTOR_IDS):
                deg = angles[i]
                target_ticks[m_id] = self.center_positions[m_id] + int(deg * self.DEG_TO_TICK)
            
            self.drive_smooth_motion(target_ticks, duration=1.6)
            time.sleep(0.4)
            
        print("[플레이백] 모든 티칭 시퀀스 재현을 종료했습니다.")
        self.listbox.selection_clear(0, tk.END)

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
        self.torque_off()
        messagebox.showwarning("비상 중지", "자동 구동이 정지되었으며, 관절의 토크가 해제되었습니다.\n로봇을 손으로 안전하게 이송할 수 있습니다.")

if __name__ == "__main__":
    root = tk.Tk()
    app = JdCobotTeachingUI(root)
    
    def on_closing():
        app.feedback_running = False
        app.stop_requested = True
        if app.is_connected:
            app.driver.close()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()