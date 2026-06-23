import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. 모델 로드 (jdcobot200.xml 명칭 사용)
xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# 탐색 설정
NUM_JOINTS = 5       # 제어할 5개 주관절
TOTAL_SAMPLES = 3000 # 몬테카를로 무작위 샘플링 개수
Z_FLOOR_LIMIT = 0.02 # 바닥 안전 마진 (m 단위)

# ==========================================
# 2. 보정 완료된 JDCobot200 표준 DH 파라미터 테이블
# [alpha, a, d, theta_offset] 구조 (Standard DH)
# ==========================================
DH_PARAMS = np.array([
    [0.0,         0.0,      0.0537,  0.0],        # Joint 1 (Base)
    [np.pi/2,     0.1352,   0.0615,  np.pi/2],    # Joint 2 (Shoulder) - 위상 보정 포함
    [0.0,         0.1352,   0.0,     0.0],        # Joint 3 (Elbow)
    [-np.pi/2,    0.0575,   0.0,     0.0],        # Joint 4 (Wrist Pitch)
    [0.0,         0.0,     -0.0310,  0.0]         # Joint 5 (Wrist Roll) + 말단 TCP 오프셋
])

# 3. Standard DH 변환 행렬 단일 계산 함수
def get_dh_matrix(alpha, a, d, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    # 교과서적인 Standard DH 단일 변환 매트릭스 형상
    T = np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0, sa,     ca,    d],
        [0.0, 0.0,    0.0,   1.0]
    ])
    return T

# 4. 순운동학(FK) 수식 기반 TCP 위치 계산 함수
def forward_kinematics_dh(joint_angles):
    T_total = np.eye(4)
    
    for i in range(NUM_JOINTS):
        alpha = DH_PARAMS[i, 0]
        a     = DH_PARAMS[i, 1]
        d     = DH_PARAMS[i, 2]
        theta_offset = DH_PARAMS[i, 3]
        
        # 현재 관절 각도에 고유 오프셋 각도 합산
        current_theta = joint_angles[i] + theta_offset
        
        # 순차적 행렬곱 연산 (T_total = T1 * T2 * ... * T5)
        T_i = get_dh_matrix(alpha, a, d, current_theta)
        T_total = np.dot(T_total, T_i)
        
    # 최종 변환 행렬에서 위치 벡터 XYZ [X, Y, Z] 추출 반환
    return T_total[:3, 3]

# 결과 데이터 저장용 리스트
safe_points = []
unsafe_points = []

# 각 관절 범위 한계 파싱
joint_ranges = [model.jnt_range[i] for i in range(NUM_JOINTS)]

# GLFW 및 시각화 초기화
glfw.init()
window = glfw.create_window(800, 600, "DH FK Safe Zone Workspace", None, None)
glfw.make_context_current(window)

scene = mj.MjvScene(model, maxgeom=5000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cam.lookat[:] = [0.0, 0.0, 0.2]
cam.distance = 1.5
cam.azimuth = 135.0
cam.elevation = -25.0

print(f"\n>>> DH 수식 기반 Safe Zone 탐색을 시작합니다 (샘플 수: {TOTAL_SAMPLES})...")

sample_count = 0

while not glfw.window_should_close(window) and sample_count < TOTAL_SAMPLES:
    # 1) 몬테카를로 무작위 각도 샘플 생성
    random_angles = np.array([np.random.uniform(j_range[0], j_range[1]) for j_range in joint_ranges])
    
    # 2) 오직 상단 'DH 수학 수식'만을 이용해 가상 TCP 예측 (물리엔진 측정값 미사용)
    tcp_pos_dh = forward_kinematics_dh(random_angles)
    
    # 3) 안전성 판단을 위해 실제 시뮬레이터에 포워딩 후 충돌 상태 확인
    data.qpos[:NUM_JOINTS] = random_angles
    mj.mj_forward(model, data)
    
    is_safe = True
    
    # 조건 A: DH 수학 수식으로 예측한 Z좌표가 바닥면 마진을 침범했는지 판단
    if tcp_pos_dh[2] < Z_FLOOR_LIMIT:
        is_safe = False
        
    # 조건 B: 해당 각도 배치 시 실제 물리 링크간 자가 충돌이 났는지 체크
    if data.ncon > 0:
        is_safe = False
        
    # 4) DH 수식 좌표 분류 기록
    if is_safe:
        safe_points.append(tcp_pos_dh)
    else:
        unsafe_points.append(tcp_pos_dh)
        
    sample_count += 1
    if sample_count % 500 == 0:
        print(f"진행 상황: {sample_count}/{TOTAL_SAMPLES} | Safe: {len(safe_points)} | Unsafe: {len(unsafe_points)}")

    # 5) 시뮬레이션 창 화면 갱신 및 실시간 렌더링
    viewport = glfw.get_framebuffer_size(window)
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(mj.MjrRect(0, 0, viewport[0], viewport[1]), scene, ctx)
    
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
print("\n>>> 탐색 완료. Matplotlib 3D 시각화 결과물을 플로팅합니다...")

# ==========================================
# 5. 수집 완료된 DH 기준 Safe Zone 3D 시각화
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

if len(safe_points) > 0:
    safe_pts = np.array(safe_points)
    ax.scatter(safe_pts[:, 0], safe_pts[:, 1], safe_pts[:, 2], 
               c='green', marker='.', s=10, alpha=0.6, label=f'DH Safe Zone ({len(safe_points)})')

if len(unsafe_points) > 0:
    unsafe_pts = np.array(unsafe_points)
    ax.scatter(unsafe_pts[:, 0], unsafe_pts[:, 1], unsafe_pts[:, 2], 
               c='red', marker='.', s=5, alpha=0.1, label=f'DH Unsafe Zone ({len(unsafe_points)})')

ax.set_xlabel('X Coordinate (m)')
ax.set_ylabel('Y Coordinate (m)')
ax.set_zlabel('Z Coordinate (m)')
ax.set_title('JDCobot200 3D Safe Zone (Calculated by DH FK)')
ax.legend(loc='upper right')

# 축 비율 균등 맞춤 처리
all_pts = np.vstack([safe_points, unsafe_points]) if len(unsafe_points) > 0 else np.array(safe_points)
max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), 
                      all_pts[:,1].max()-all_pts[:,1].min(), 
                      all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0

mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

output_filename = 'dh_safe_zone_workspace.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=150)
plt.close()

print(f">>> 성공적으로 DH 기준 세이프존 이미지가 '{output_filename}' 파일로 빌드되었습니다.")