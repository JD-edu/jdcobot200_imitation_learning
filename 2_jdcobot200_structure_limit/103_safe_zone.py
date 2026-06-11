import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =====================================================================
# 1. 하드웨어 치수 및 안전 구역 가이드라인 (단위: 미터)
# =====================================================================
L1, L2, L3, L4 = 0.080, 0.120, 0.120, 0.060

# 베이스 충돌 감지용 원통 설정
BASE_RADIUS = 0.065  # 베이스 반경 (6.5cm)
BASE_HEIGHT = 0.090  # 베이스 높이 (9cm)
GROUND_LEVEL = 0.0   # 지면 높이 한계 (0cm, 책상 바닥)

# =====================================================================
# 2. 정방향 기구학 (모든 조인트 및 링크 엔드포인트 좌표 도출)
# =====================================================================
def get_joint_positions(q1, q2, q3, q4):
    p0 = np.array([0, 0, 0])      # 바닥 원점
    p1 = np.array([0, 0, L1])     # Shoulder (J2)
    
    ang2 = q2
    ang3 = q2 + q3
    ang4 = q2 + q3 + q4
    
    # Elbow (J3) 위치
    r2 = L2 * np.cos(ang2)
    p2 = np.array([r2 * np.cos(q1), r2 * np.sin(q1), L1 + L2 * np.sin(ang2)])
    
    # Wrist (J4) 위치
    r3 = r2 + L3 * np.cos(ang3)
    p3 = np.array([r3 * np.cos(q1), r3 * np.sin(q1), p2[2] + L3 * np.sin(ang3)])
    
    # End-Effector (손끝 팁) 위치
    r4 = r3 + L4 * np.cos(ang4)
    p4 = np.array([r4 * np.cos(q1), r4 * np.sin(q1), p3[2] + L4 * np.sin(ang4)])
    
    return p0, p1, p2, p3, p4

# 베이스 원통 시각화용 그리드 데이터 생성
def generate_cylinder(radius, height):
    z = np.linspace(0, height, 10)
    theta = np.linspace(0, 2*np.pi, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    return x_grid, y_grid, z_grid

# =====================================================================
# 3. 메인 인터락(Interlock) 및 시각화 빌드
# =====================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.3)

# 로봇이 바로 직전에 위치했던 '안전한 각도'를 기억하는 백업 저장소
# 위험 지역 진입 시 이 각도로 강제 롤백시킵니다.
last_safe_angles = [0.0, np.deg2rad(45), np.deg2rad(-45), 0.0]

# UI 객체 초기화
robot_line, = ax.plot([], [], [], 'ko-', lw=3, ms=6, zorder=5)
cylinder_container = [None]

# 실시간 감시 및 제어 루프
def check_safety_and_update(val):
    global last_safe_angles
    
    # 1) 사용자가 슬라이더로 조작한 현재 각도 읽기
    q1 = s_q1.val
    q2 = s_q2.val
    q3 = s_q3.val
    q4 = s_q4.val
    
    # 2) 입력된 각도로부터 로봇의 모든 마디 위치 계산
    p0, p1, p2, p3, p4 = get_joint_positions(q1, q2, q3, q4)
    
    # 3) 실시간 안전 진단 (Safety Audit)
    is_safe = True
    alert_msg = ""
    
    # 검사 A: 지면 충돌 체크 (모든 조인트의 Z 좌표가 바닥보다 밑으로 내려갔는가?)
    if p2[2] < GROUND_LEVEL or p3[2] < GROUND_LEVEL or p4[2] < GROUND_LEVEL:
        is_safe = False
        alert_msg = "⛔ [지면 충돌 위험] 로봇이 바닥(책상)을 뚫고 가려 합니다!"
        
    # 검사 B: 자가 충돌 체크 (손끝 p4가 베이스 원통 가상 한계 반경/높이 내부에 갇혔는가?)
    ee_r = np.sqrt(p4[0]**2 + p4[1]**2)
    ee_z = p4[2]
    if ee_r < BASE_RADIUS and ee_z < BASE_HEIGHT:
        is_safe = False
        alert_msg = "⛔ [자가 충돌 위험] 손끝이 베이스 바디를 타격하려 합니다!"

    # 4) 진단 결과에 따른 하드웨어 보호 인터락 가동
    if not is_safe:
        # [핵심] 위험 구역으로 가는 슬라이더 마우스 조작을 무시하고, 이전 안전 각도로 강제 복귀!
        s_q1.eventson = s_q2.eventson = s_q3.eventson = s_q4.eventson = False # 무한 루프 방지용 이벤트 off
        s_q1.set_val(last_safe_angles[0])
        s_q2.set_val(last_safe_angles[1])
        s_q3.set_val(last_safe_angles[2])
        s_q4.set_val(last_safe_angles[3])
        s_q1.eventson = s_q2.eventson = s_q3.eventson = s_q4.eventson = True  # 이벤트 재가동
        
        # 다시 안전한 각도의 위치로 좌표 재계산
        p0, p1, p2, p3, p4 = get_joint_positions(*last_safe_angles)
        body_color = 'red'
        ax.set_title(alert_msg, color='red', fontsize=12, weight='bold')
    else:
        # 안전한 구역에 안착했다면, 현재 각도를 새로운 '최신 안전 각도'로 백업
        last_safe_angles = [q1, q2, q3, q4]
        body_color = 'green'
        ax.set_title("🟢 [안전] 인터락 작동 중 - 구동 가능 영역", color='green', fontsize=12)

    # 5) 화면 갱신
    robot_line.set_data([p0[0], p1[0], p2[0], p3[0], p4[0]], [p0[1], p1[1], p2[1], p3[1], p4[1]])
    robot_line.set_3d_properties([p0[2], p1[2], p2[2], p3[2], p4[2]])
    
    if cylinder_container[0] is not None:
        cylinder_container[0].remove()
    cx, cy, cz = generate_cylinder(BASE_RADIUS, BASE_HEIGHT)
    cylinder_container[0] = ax.plot_surface(cx, cy, cz, color=body_color, alpha=0.3, edgecolor='none')
    
    fig.canvas.draw_idle()

# =====================================================================
# 4. GUI 슬라이더 인터페이스 배치 (가동범위 제한 포함)
# =====================================================================
ax_q1 = plt.axes([0.2, 0.20, 0.6, 0.025])
ax_q2 = plt.axes([0.2, 0.15, 0.6, 0.025])
ax_q3 = plt.axes([0.2, 0.10, 0.6, 0.025])
ax_q4 = plt.axes([0.2, 0.05, 0.6, 0.025])

# 슬라이더 가동 영역 자체를 하드웨어 제한(Joint Limit) 값으로 1차 고정
s_q1 = Slider(ax_q1, 'J1 (Base)', -np.deg2rad(150), np.deg2rad(150), valinit=last_safe_angles[0])
s_q2 = Slider(ax_q2, 'J2 (Shoulder)', -np.deg2rad(60), np.deg2rad(90), valinit=last_safe_angles[1])
s_q3 = Slider(ax_q3, 'J3 (Elbow)', -np.deg2rad(90), np.deg2rad(90), valinit=last_safe_angles[2])
s_q4 = Slider(ax_q4, 'J4 (Wrist)', -np.deg2rad(90), np.deg2rad(90), valinit=last_safe_angles[3])

s_q1.on_changed(check_safety_and_update)
s_q2.on_changed(check_safety_and_update)
s_q3.on_changed(check_safety_and_update)
s_q4.on_changed(check_safety_and_update)

# 3D 뷰포트 고정 및 연동
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.25, 0.25)
ax.set_zlim(0, 0.35)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.view_init(elev=20, azim=45)

# 가상 바닥(지면) 그리드 레이아웃 표현
gx, gy = np.meshgrid(np.linspace(-0.25, 0.25, 10), np.linspace(-0.25, 0.25, 10))
ax.plot_wireframe(gx, gy, np.zeros_like(gx), color='gray', alpha=0.2, linestyle=':')

check_safety_and_update(None)
plt.show()