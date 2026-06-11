import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =====================================================================
# 1. 로봇 치수 및 안전 반경 설정 (단위: 미터)
# =====================================================================
L1, L2, L3, L4 = 0.080, 0.120, 0.120, 0.060

# 산업용 단순화 방식을 적용한 베이스 충돌 원통 치수
BASE_RADIUS = 0.060  # 베이스 몸체 반경 (6cm)
BASE_HEIGHT = 0.090  # 베이스 몸체 높이 (9cm)

# =====================================================================
# 2. 순방향 기구학 및 관절 위치 도출 함수
# =====================================================================
def get_joint_positions(q1, q2, q3, q4):
    """ 각 조인트의 3D 공간 좌표 (x, y, z)를 차례대로 반환 """
    # Base (J1) 위치
    p0 = np.array([0, 0, 0])
    
    # Shoulder (J2) 위치
    p1 = np.array([0, 0, L1])
    
    # 수직 단면 상의 누적 각도 계산
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

# =====================================================================
# 3. 가상 원통형 바디(Cylinder Mesh) 생성 함수
# =====================================================================
def generate_cylinder(radius, height, z_offset=0):
    """ 중심축이 Z축인 원기둥의 X, Y, Z 메쉬 그리드 데이터 생성 """
    z = np.linspace(0, height, 10) + z_offset
    theta = np.linspace(0, 2*np.pi, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    return x_grid, y_grid, z_grid

# =====================================================================
# 4. 메인 GUI 및 시각화 빌드
# =====================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.3) # 슬라이더가 들어갈 하단 공간 확보

# 초기 각도 (라디안)
q1_init, q2_init, q3_init, q4_init = 0.0, np.deg2rad(30), np.deg2rad(-45), 0.0

# 뼈대 3D 선(Line) 객체와 원통(Surface) 객체 초기화용 변수
robot_line, = ax.plot([], [], [], 'ko-', lw=3, ms=6, zorder=5)
cylinder_surf = [None]

def update_plot(val):
    global cylinder_surf
    
    # 슬라이더로부터 현재 관절 값 읽기
    q1 = s_q1.val
    q2 = s_q2.val
    q3 = s_q3.val
    q4 = s_q4.val
    
    # 각 관절의 현재 3D 좌표 계산
    p0, p1, p2, p3, p4 = get_joint_positions(q1, q2, q3, q4)
    
    # 1) 뼈대 라인 업데이트
    xs = [p0[0], p1[0], p2[0], p3[0], p4[0]]
    ys = [p0[1], p1[1], p2[1], p3[1], p4[1]]
    zs = [p0[2], p1[2], p2[2], p3[2], p4[2]]
    robot_line.set_data(xs, ys)
    robot_line.set_3d_properties(zs)
    
    # 2) [충돌 검사 알고리즘] 손끝(p4)이 베이스 원통 범위 안에 파고들었는지 체크
    # 손끝의 평면 중심거리(R) 계산
    ee_r = np.sqrt(p4[0]**2 + p4[1]**2)
    ee_z = p4[2]
    
    # 베이스 원통 반경 안쪽에 있고, 높이도 베이스 윗면보다 아래에 있다면 충돌!
    if ee_r < BASE_RADIUS and ee_z < BASE_HEIGHT:
        body_color = 'red'  # 위험 상황 시 빨간색 원통으로 변경
        ax.set_title("⚠️ [COLLISION DETECTED] 손끝이 베이스를 치고 있습니다!", color='red', fontsize=14, weight='bold')
    else:
        body_color = 'cyan' # 안전 상황 시 푸른색 원통
        ax.set_title("jdcobot200 Cylinder Safety Simulator", color='black', fontsize=12)
        
    # 3) 가상 원통 바디 다시 그리기 (기존 원통 제거 후 갱신)
    if cylinder_surf[0] is not None:
        cylinder_surf[0].remove()
        
    cx, cy, cz = generate_cylinder(BASE_RADIUS, BASE_HEIGHT, z_offset=0)
    cylinder_surf[0] = ax.plot_surface(cx, cy, cz, color=body_color, alpha=0.4, edgecolor='none')
    
    fig.canvas.draw_idle()

# =====================================================================
# 5. 하단 인터페이스 슬라이더 배치
# =====================================================================
ax_q1 = plt.axes([0.2, 0.20, 0.6, 0.03])
ax_q2 = plt.axes([0.2, 0.15, 0.6, 0.03])
ax_q3 = plt.axes([0.2, 0.10, 0.6, 0.03])
ax_q4 = plt.axes([0.2, 0.05, 0.6, 0.03])

s_q1 = Slider(ax_q1, 'J1 (Base)', -np.pi, np.pi, valinit=q1_init, valfmt='%1.2f rad')
s_q2 = Slider(ax_q2, 'J2 (Shoulder)', -np.pi/2, np.pi/2, valinit=q2_init, valfmt='%1.2f rad')
s_q3 = Slider(ax_q3, 'J3 (Elbow)', -np.pi/2, np.pi/2, valinit=q3_init, valfmt='%1.2f rad')
s_q4 = Slider(ax_q4, 'J4 (Wrist)', -np.pi/2, np.pi/2, valinit=q4_init, valfmt='%1.2f rad')

# 슬라이더 값이 바뀔 때마다 update_plot 함수 실행 연동
s_q1.on_changed(update_plot)
s_q2.on_changed(update_plot)
s_q3.on_changed(update_plot)
s_q4.on_changed(update_plot)

# =====================================================================
# 6. 시뮬레이션 환경 초기화 및 축 레이아웃 고정
# =====================================================================
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.25, 0.25)
ax.set_zlim(0, 0.35)
ax.view_init(elev=20, azim=45)

# 첫 화면 구동
update_plot(None)
plt.show()