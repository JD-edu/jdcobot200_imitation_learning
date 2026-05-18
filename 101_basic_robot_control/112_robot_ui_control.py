import time
import math
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from motor_control import MiniFeetechDriver

class JdCobotUI:
    def __init__(self, window):
        self.window = window
        self.window.title("jdcobot200 통합 제어 시스템")
        self.window.geometry("850x650")
        
        # --- 로봇 연결 및 하드웨어 설정 ---
        self.PORT = "/dev/ttyUSB0"
        self.BAUDRATE = 1000000
        self.MOTOR_IDS = [1, 2, 3, 4, 5, 6]
        self.DEG_TO_TICK = 4096.0 / 360.0
        
        try:
            self.driver = MiniFeetechDriver(self.PORT, self.BAUDRATE)
            self.is_connected = True
        except Exception as e:
            print(f"로봇 연결 실패: {e}")
            self.is_connected = False
            messagebox.showerror("연결 오류", f"로봇을 연결할 수 없습니다. 포트를 확인하세요.\n({e})")

        # --- 상태 및 제어 변수 ---
        self.center_positions = {}   # 칼리브레이션 기준 위치 (원점)
        self.current_positions = {}  # 현재 가상/실제 목표 위치 (Tick)
        self.saved_waypoints = []    # 저장된 목표점 리스트
        self.is_moving = False       # 현재 모션 실행 중 여부
        self.stop_requested = False  # 비상 정지 플래그
        
        # 원점 초기값 설정 및 하드웨어 가감속 세팅
        if self.is_connected:
            for m_id in self.MOTOR_IDS:
                pos = self.driver.get_position_filtered(m_id, samples=5)
                if pos is None: pos = 2048 # 실패 시 센터 기본값
                self.center_positions[m_id] = pos
                self.current_positions[m_id] = pos
                self.driver.set_torque(m_id, True)
                
                # 하드웨어 가속도/속도 사전 제한 (진동 저감)
                if hasattr(self.driver, 'set_acceleration'): self.driver.set_acceleration(m_id, 40)
                if hasattr(self.driver, 'set_speed'): self.driver.set_speed(m_id, 1200)
        else:
            # 시뮬레이션 모드용 기본값
            for m_id in self.MOTOR_IDS:
                self.center_positions[m_id] = 2048
                self.current_positions[m_id] = 2048

        # --- UI 레이아웃 구성 ---
        self.create_widgets()

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
        
        left_frame = tk.LabelFrame(main_frame, text=" 관절 개별 제어 (1번~6번) ", padx=15, pady=15)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_frame = tk.LabelFrame(main_frame, text=" 3~4. 연속 이동 시퀀스 저장소 ", padx=15, pady=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)

        # --- 좌측: 1 & 2번 요구사항 기능 구현 ---
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
            
            # 2. 개별 제어 슬라이더 (가동 범위 -150도 ~ 150도 가정)
            slider = ttk.Scale(row_frame, from_=-150, to=150, orient=tk.HORIZONTAL, value=0, command=lambda val, m=m_id: self.slider_moved(m, val))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            self.sliders[m_id] = slider
            
            lbl_ang = tk.Label(row_frame, text="0.0°", width=8, anchor='e')
            lbl_ang.pack(side=tk.LEFT)
            self.lbl_angles[m_id] = lbl_ang

        # 2번 항목: 전체 동시 이동 버튼
        self.btn_move_all = tk.Button(left_frame, text="▶ 설정된 슬라이더 위치로 동시 이동", bg="#3498db", fg="white", font=('Arial', 11, 'bold'), pady=8, command=self.move_to_sliders)
        self.btn_move_all.pack(fill=tk.X, pady=(20, 15))

        # --- 우측: 3 & 4번 요구사항 기능 구현 ---
        # 연속 이동점 표시 리스트박스
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
        """
        [사인파 가감속 추종 알고리즘]
        target_ticks: 각 모터 ID별 가고 싶은 목표 Tick 딕셔너리
        """
        if not self.is_connected or self.stop_requested:
            # 하드웨어 미연결 시 위치 변수만 업데이트 시뮬레이션
            for m_id, t_tick in target_ticks.items():
                self.current_positions[m_id] = t_tick
            return

        self.is_moving = True
        start_ticks = {m_id: self.current_positions[m_id] for m_id in self.MOTOR_IDS}
        
        control_period = 0.02  # 20ms 제어 주기
        steps = int(duration / control_period)
        
        for step in range(steps + 1):
            if self.stop_requested:
                break
                
            # Sine 곡선 가감속 비율 계산 (0.0 ~ 1.0)
            ratio = (1.0 - math.cos((step / steps) * math.pi)) / 2.0
            
            for m_id in self.MOTOR_IDS:
                if m_id in target_ticks:
                    current_target = start_ticks[m_id] + (target_ticks[m_id] - start_ticks[m_id]) * ratio
                    current_target = max(0, min(4095, int(current_target)))
                    self.driver.set_position(m_id, current_target)
                    self.current_positions[m_id] = current_target
                    
            time.sleep(control_period)
        self.is_moving = False

    # --- 각 기능별 서브루틴 및 이벤트 핸들러 ---
    def jog_joint(self, motor_id, deg_delta):
        """ 1. 버튼 누르면 단일 관절 3도씩 좌우 구동 """
        if self.is_moving: return
        
        # 현재 지정된 슬라이더 각도 값을 읽어 이동 계산
        current_deg = self.sliders[motor_id].get()
        new_deg = max(-150.0, min(150.0, current_deg + deg_delta))
        
        # UI 동기화
        self.sliders[motor_id].set(new_deg)
        self.lbl_angles[motor_id].config(text=f"{new_deg:+.1f}°")
        
        # 목표 틱 계산 후 구동 명령
        target_tick = self.center_positions[motor_id] + int(new_deg * self.DEG_TO_TICK)
        target_dict = {motor_id: target_tick}
        
        threading.Thread(target=self.drive_smooth_motion, args=(target_dict, 0.4), daemon=True).start()

    def slider_moved(self, motor_id, value):
        """ 슬라이더 조작 시 텍스트 라벨 실시간 변경 """
        val = float(value)
        self.lbl_angles[motor_id].config(text=f"{val:+.1f}°")

    def move_to_sliders(self):
        """ 2. 이동 버튼을 누르면 모든 관절이 타겟 지점으로 동시에 부드럽게 이동 """
        if self.is_moving: return
        self.stop_requested = False
        
        target_ticks = {}
        for m_id in self.MOTOR_IDS:
            deg = self.sliders[m_id].get()
            target_ticks[m_id] = self.center_positions[m_id] + int(deg * self.DEG_TO_TICK)
            
        threading.Thread(target=self.drive_smooth_motion, args=(target_ticks, 1.5), daemon=True).start()

    def save_current_position(self):
        """ 3. 현재 각 관절의 목표 각도 상태를 시퀀스 리스트에 저장 """
        current_angles = [self.sliders[m_id].get() for m_id in self.MOTOR_IDS]
        self.saved_waypoints.append(current_angles)
        
        # 리스트박스 문자열 포맷팅 추가
        idx = len(self.saved_waypoints)
        ang_str = ", ".join([f"{a:+.1f}°" for a in current_angles])
        self.listbox.insert(tk.END, f"P{idx:02d} -> [{ang_str}]")

    def clear_waypoints(self):
        """ 저장된 리스트 초기화 """
        self.saved_waypoints.clear()
        self.listbox.delete(0, tk.END)

    def start_sequence_thread(self):
        """ 4. 연속 이동 스레드 기동 """
        if self.is_moving: return
        if not self.saved_waypoints:
            messagebox.showwarning("경고", "저장된 목표점이 없습니다. 먼저 위치를 저장하세요.")
            return
        
        self.stop_requested = False
        threading.Thread(target=self.run_sequence, daemon=True).start()

    def run_sequence(self):
        """ 저장된 연속 이동점 순서대로 로봇암 구동 루틴 """
        print("[시퀀스] 연속 이동을 시작합니다.")
        for idx, angles in enumerate(self.saved_waypoints):
            if self.stop_requested:
                break
                
            print(f"[시퀀스] {idx+1}번째 포인트로 가감속 이동 중...")
            # 리스트박스 하이라이트 선택 효과
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            
            # 슬라이더 값 업데이트 및 목표 틱 변환
            target_ticks = {}
            for i, m_id in enumerate(self.MOTOR_IDS):
                deg = angles[i]
                self.sliders[m_id].set(deg)
                self.lbl_angles[m_id].config(text=f"{deg:+.1f}°")
                target_ticks[m_id] = self.center_positions[m_id] + int(deg * self.DEG_TO_TICK)
            
            # 이동 실행 후 완전히 도착할 때까지 블로킹 대기 후 다음 모션 진행
            self.drive_smooth_motion(target_ticks, duration=1.8)
            time.sleep(0.5) # 포인트 도달 후 안정화 대기 시간
        print("[시퀀스] 연속 이동이 완료되었습니다.")

    def go_home(self):
        """ 5-1. 모든 동작 취소 및 홈(원점) 위치 복귀 """
        self.stop_requested = True  # 기존 진행 중인 시퀀스 파괴
        time.sleep(0.05)            # 스레드 종료 대기 간격
        self.stop_requested = False
        
        print("[홈] 모든 동작을 취소하고 칼리브레이션 원점으로 복귀합니다.")
        # UI 초기화
        for m_id in self.MOTOR_IDS:
            self.sliders[m_id].set(0.0)
            self.lbl_angles[m_id].config(text="0.0°")
            
        # 가감속 원점 복귀 트랙션 실행
        threading.Thread(target=self.drive_smooth_motion, args=(self.center_positions, 2.0), daemon=True).start()

    def emergency_stop(self):
        """ 5-2. 비상 중지: 모든 모션 중단 및 즉시 토크 해제(Freemode) """
        self.stop_requested = True
        print("\n🚨🚨🚨 비상 중지(EMERGENCY STOP)가 발동되었습니다! 🚨🚨🚨")
        
        if self.is_connected:
            for m_id in self.MOTOR_IDS:
                self.driver.set_torque(m_id, False) # 토크 해제하여 축 늘어뜨리기 (안전조치)
        
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