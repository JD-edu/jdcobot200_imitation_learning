import time
import math
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from motor_control import MiniFeetechDriver

class JdCobotUI:
    def __init__(self, window):
        self.window = window
        self.window.title("jdcobot200 통합 제어 시스템 (오프셋/리미트 반영)")
        self.window.geometry("850x650")
        
        # --- 로봇 연결 및 하드웨어 설정 ---
        self.PORT = "/dev/ttyACM0"
        self.BAUDRATE = 1000000
        self.MOTOR_IDS = [1, 2, 3, 4, 5, 6]
        self.DEG_TO_TICK = 4096.0 / 360.0
        self.TICK_TO_DEG = 360.0 / 4096.0
        self.THEORETICAL_CENTER = 2048 # STS3215의 이론상 물리적 중심점
        
        # --- 파일 로드 데이터 공간 ---
        self.offsets = {m_id: 0 for m_id in self.MOTOR_IDS}
        self.angle_limits = {m_id: (-150.0, 150.0) for m_id in self.MOTOR_IDS} # 기본값 보장
        
        # 파일 데이터 파싱 우선 진행
        self.load_offsets_from_file()
        self.load_limits_from_file()
        
        try:
            self.driver = MiniFeetechDriver(self.PORT, self.BAUDRATE)
            self.is_connected = True
        except Exception as e:
            print(f"로봇 연결 실패: {e}")
            self.is_connected = False
            messagebox.showerror("연결 오류", f"로봇을 연결할 수 없습니다. 포트를 확인하세요.\n({e})")

        # --- 상태 및 제어 변수 ---
        self.center_positions = {}   # 칼리브레이션 기준 위치 (2048 + 오프셋)
        self.current_positions = {}  # 현재 가상/실제 목표 위치 (Tick)
        self.saved_waypoints = []    # 저장된 목표점 리스트
        self.is_moving = False       # 현재 모션 실행 중 여부
        self.stop_requested = False  # 비상 정지 플래그
        
        # 원점 보정 적용 및 하드웨어 가감속 세팅
        for m_id in self.MOTOR_IDS:
            # 소프트웨어 보정 방식: 2048 중심점에 파일에서 읽은 오프셋 적용
            software_home = self.THEORETICAL_CENTER + self.offsets[m_id]
            self.center_positions[m_id] = software_home
            self.current_positions[m_id] = software_home
            
            if self.is_connected:
                self.driver.set_torque(m_id, True)
                # 하드웨어 가속도/속도 사전 제한 (진동 저감)
                if hasattr(self.driver, 'set_acceleration'): self.driver.set_acceleration(m_id, 40)
                if hasattr(self.driver, 'set_speed'): self.driver.set_speed(m_id, 1200)

        # --- UI 레이아웃 구성 ---
        self.create_widgets()

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
            print(f"▶ [{file_path}] 오프셋 데이터 성공적 로드: {self.offsets}")
        except Exception as e:
            print(f"❌ 오프셋 파일 로드 중 오류 발생: {e}")

    def load_limits_from_file(self, file_path="joint_limits.txt"):
        """ 
        텍스트 파일에서 측정 리미트 데이터를 로드하고,
        오프셋이 반영된 기준점에서 안전 버퍼 5도를 차감/가산한 뒤 5단위로 버림/올림하여 각도 한계계를 계산합니다.
        """
        if not os.path.exists(file_path):
            print(f"⚠️ [{file_path}] 파일이 없어 기본 가동범위(-150 ~ 150)를 사용합니다.")
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    if "=" in line:
                        motor_id, limit_vals = line.split("=")
                        m_id = int(motor_id)
                        if m_id in self.MOTOR_IDS:
                            # 예: "1=1100,3100" 형태 파싱
                            min_tick, max_tick = map(int, limit_vals.split(","))
                            
                            # 1단계: 수동 측정 틱(오프셋 반영본)을 소프트웨어 원점 기준 상대 각도로 변환
                            home_tick = self.THEORETICAL_CENTER + self.offsets[m_id]
                            min_deg_raw = (min_tick - home_tick) * self.TICK_TO_DEG
                            max_deg_raw = (max_tick - home_tick) * self.TICK_TO_DEG
                            
                            # 2단계: 양 끝단에 5도 마진(버퍼) 적용
                            min_deg_buffered = min_deg_raw + 5.0
                            max_deg_buffered = max_deg_raw - 5.0
                            
                            # 3단계: 5도 단위로 정밀하게 끊기 (최소값은 올림 처리, 최대값은 내림 처리로 가동범위 축소 안전화)
                            safe_min_deg = math.ceil(min_deg_buffered / 5.0) * 5.0
                            safe_max_deg = math.floor(max_deg_buffered / 5.0) * 5.0
                            
                            # 최종 제한 매핑 보관
                            self.angle_limits[m_id] = (safe_min_deg, safe_max_deg)
            print(f"▶ [{file_path}] 안전 마진(5° 버퍼, 5단위 정렬)이 반영된 가동 범위 한계:")
            for m_id, limits in self.angle_limits.items():
                print(f"   - 관절 {m_id}: {limits[0]}° ~ {limits[1]}°")
        except Exception as e:
            print(f"❌ 리미트 파일 로드 중 오류 발생: {e}")

    def create_widgets(self):
        # 상단 제어 버튼부 (홈 / 중지)
        top_frame = tk.Frame(self.window, pady=10)
        top_frame.pack(fill=tk.X)
        
        self.btn_home = tk.Button(top_frame, text="🏠 홈 위치 이동", bg="#2ecc71", fg="white", font=('Arial', 11, 'bold'), width=15, command=self.go_home)
        self.btn_home.pack(side=tk.LEFT, padx=20)
        
        self.btn_stop = tk.Button(top_frame, text="🚨 비상 중지 (토크 오프)", bg="#e74c3c", fg="white", font=('Arial', 11, 'bold'), width=22, command=self.emergency_stop)
        self.btn_stop.pack(side=tk.RIGHT, padx=20)

        # 메인 콘텐츠 영역 (좌측: 조작부, 우측: 시퀀스 리스트)
        main_frame = tk.Frame(self.window, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.LabelFrame(main_frame, text=" 관절 개별 제어 (안전 소프트웨어 리미트 적용) ", padx=15, pady=15)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_frame = tk.LabelFrame(main_frame, text=" 3~4. 연속 이동 시퀀스 저장소 ", padx=15, pady=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)

        # --- 좌측: 관절별 동적 리미트 슬라이더 및 UI 빌드 ---
        self.sliders = {}
        self.lbl_angles = {}
        
        for idx, m_id in enumerate(self.MOTOR_IDS):
            row_frame = tk.Frame(left_frame, pady=6)
            row_frame.pack(fill=tk.X)
            
            lbl_name = tk.Label(row_frame, text=f"관절 {m_id}:", font=('Arial', 10, 'bold'), width=6, anchor='w')
            lbl_name.pack(side=tk.LEFT)
            
            # 1. 3도 증감 버튼
            btn_dec = tk.Button(row_frame, text="-3°", width=4, command=lambda m=m_id: self.jog_joint(m, -3))
            btn_dec.pack(side=tk.LEFT, padx=2)
            
            btn_inc = tk.Button(row_frame, text="+3°", width=4, command=lambda m=m_id: self.jog_joint(m, 3))
            btn_inc.pack(side=tk.LEFT, padx=2)
            
            # 파일에서 안전 연산 처리가 완료된 제한범위를 슬라이더 스케일에 바인딩 (from_, to)
            min_lim, max_lim = self.angle_limits[m_id]
            
            slider = ttk.Scale(row_frame, from_=min_lim, to=max_lim, orient=tk.HORIZONTAL, value=0, command=lambda val, m=m_id: self.slider_moved(m, val))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            self.sliders[m_id] = slider
            
            lbl_ang = tk.Label(row_frame, text="0.0°", width=8, anchor='e')
            lbl_ang.pack(side=tk.LEFT)
            self.lbl_angles[m_id] = lbl_ang

        # 전체 동시 이동 버튼
        self.btn_move_all = tk.Button(left_frame, text="▶ 설정된 슬라이더 위치로 동시 이동", bg="#3498db", fg="white", font=('Arial', 11, 'bold'), pady=8, command=self.move_to_sliders)
        self.btn_move_all.pack(fill=tk.X, pady=(20, 15))

        # --- 우측: 연속 이동 시퀀스 저장소 ---
        self.listbox = tk.Listbox(right_frame, width=35, height=18, font=('Courier', 9))
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        btn_save = tk.Button(right_frame, text="💾 현재 위치 저장", bg="#9b59b6", fg="white", font=('Arial', 10), command=self.save_current_position)
        btn_save.pack(fill=tk.X, pady=3)
        
        btn_clear = tk.Button(right_frame, text="🗑️ 리스트 비우기", bg="#7f8c8d", fg="white", font=('Arial', 10), command=self.clear_waypoints)
        btn_clear.pack(fill=tk.X, pady=3)
        
        self.btn_play = tk.Button(right_frame, text="🔁 연속 이동 실행", bg="#f1c40f", fg="black", font=('Arial', 11, 'bold'), pady=5, command=self.start_sequence_thread)
        self.btn_play.pack(fill=tk.X, pady=8)

    # --- 실시간 제어 핵심 알고리즘 ---
    def drive_smooth_motion(self, target_ticks, duration=1.5):
        if not self.is_connected or self.stop_requested:
            for m_id, t_tick in target_ticks.items():
                self.current_positions[m_id] = t_tick
            return

        self.is_moving = True
        start_ticks = {m_id: self.current_positions[m_id] for m_id in self.MOTOR_IDS}
        
        control_period = 0.02  # 20ms 제어 주기 [cite: 52, 54]
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
                    time.sleep(0.0015)  # 명령 패킷 연속 충돌 방지를 위한 초미세 가이드 지연
            time.sleep(control_period)
        self.is_moving = False

    def jog_joint(self, motor_id, deg_delta):
        """ 버튼 누르면 단일 관절 3도씩 제한 범위 내에서 안전 이동 """
        if self.is_moving: return
        
        current_deg = self.sliders[motor_id].get()
        min_lim, max_lim = self.angle_limits[motor_id]
        new_deg = max(min_lim, min(max_lim, current_deg + deg_delta))
        
        # UI 동기화
        self.sliders[motor_id].set(new_deg)
        self.lbl_angles[motor_id].config(text=f"{new_deg:+.1f}°")
        
        # 오프셋 반영 원점 좌표에 각도 오차를 합산하여 타겟 틱 연산
        target_tick = self.center_positions[motor_id] + int(new_deg * self.DEG_TO_TICK)
        target_dict = {motor_id: target_tick}
        
        threading.Thread(target=self.drive_smooth_motion, args=(target_dict, 0.4), daemon=True).start()

    def slider_moved(self, motor_id, value):
        val = float(value)
        self.lbl_angles[motor_id].config(text=f"{val:+.1f}°")

    def move_to_sliders(self):
        """ 모든 관절이 안전 한계가 적용된 타겟 지점으로 동시에 부드럽게 이동 """
        if self.is_moving: return
        self.stop_requested = False
        
        target_ticks = {}
        for m_id in self.MOTOR_IDS:
            deg = self.sliders[m_id].get()
            target_ticks[m_id] = self.center_positions[m_id] + int(deg * self.DEG_TO_TICK)
            
        threading.Thread(target=self.drive_smooth_motion, args=(target_ticks, 1.5), daemon=True).start()

    def save_current_position(self):
        current_angles = [self.sliders[m_id].get() for m_id in self.MOTOR_IDS]
        self.saved_waypoints.append(current_angles)
        
        idx = len(self.saved_waypoints)
        ang_str = ", ".join([f"{a:+.1f}°" for a in current_angles])
        self.listbox.insert(tk.END, f"P{idx:02d} -> [{ang_str}]")

    def clear_waypoints(self):
        self.saved_waypoints.clear()
        self.listbox.delete(0, tk.END)

    def start_sequence_thread(self):
        if self.is_moving: return
        if not self.saved_waypoints:
            messagebox.showwarning("경고", "저장된 목표점이 없습니다. 먼저 위치를 저장하세요.")
            return
        
        self.stop_requested = False
        threading.Thread(target=self.run_sequence, daemon=True).start()

    def run_sequence(self):
        print("[시퀀스] 연속 이동을 시작합니다.")
        for idx, angles in enumerate(self.saved_waypoints):
            if self.stop_requested:
                break
                
            print(f"[시퀀스] {idx+1}번째 포인트로 가감속 이동 중...")
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            
            target_ticks = {}
            for i, m_id in enumerate(self.MOTOR_IDS):
                deg = angles[i]
                self.sliders[m_id].set(deg)
                self.lbl_angles[m_id].config(text=f"{deg:+.1f}°")
                target_ticks[m_id] = self.center_positions[m_id] + int(deg * self.DEG_TO_TICK)
            
            self.drive_smooth_motion(target_ticks, duration=1.8)
            time.sleep(0.5)
        print("[시퀀스] 연속 이동이 완료되었습니다.")

    def go_home(self):
        """ 모든 동작을 취소하고 로드된 캘리브레이션 소프트웨어 원점으로 정렬 """
        self.stop_requested = True
        time.sleep(0.05)
        self.stop_requested = False
        
        print("[홈] 모든 동작을 취소하고 파일 오프셋이 적용된 원점으로 복귀합니다.")
        for m_id in self.MOTOR_IDS:
            self.sliders[m_id].set(0.0)
            self.lbl_angles[m_id].config(text="0.0°")
            
        threading.Thread(target=self.drive_smooth_motion, args=(self.center_positions, 2.0), daemon=True).start()

    def emergency_stop(self):
        self.stop_requested = True
        print("\n🚨🚨🚨 비상 중지(EMERGENCY STOP)가 발동되었습니다! 🚨🚨🚨")
        
        if self.is_connected:
            for m_id in self.MOTOR_IDS:
                self.driver.set_torque(m_id, False)
        
        messagebox.showerror("비상 중지", "비상 중지가 클릭되었습니다.\n모든 모터의 토크를 해제했습니다.\n다시 작동하려면 프로그램을 재시작하세요.")

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