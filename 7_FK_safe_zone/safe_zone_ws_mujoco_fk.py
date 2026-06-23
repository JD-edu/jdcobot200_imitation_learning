import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. 모델 및 데이터 로드 (jdcobot200.xml 명칭에 맞춤)
xml_path = os.path.join(os.path.dirname(__file__), "jdcobot200.xml") # 파일명 확인 필수
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# 탐색 설정
NUM_JOINTS = 5       # 그리퍼 제외 5축 제어
TOTAL_SAMPLES = 3000 # 무작위 샘플링 개수
Z_FLOOR_LIMIT = 0.02 # 바닥 안전 마진 (m 단위)

# ==========================================
# 안전한 TCP ID 추출 매커니즘 (에러 방지)
# ==========================================
tcp_id = -1
use_site = False

# 1) 먼저 "tcp" 이름을 가진 site가 있는지 검색
try:
    tcp_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "gripper_base")
    if tcp_id != -1:
        use_site = True
        print(f"[안내] 'tcp' 사이트(ID: {tcp_id})를 찾아내어 말단 좌표로 설정합니다.")
except:
    pass

# 2) site가 없다면 마지막 바디(Link 5 혹은 그리퍼 베이스)를 자동으로 타깃팅
if tcp_id == -1:
    tcp_id = model.nbody - 1
    use_site = False
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, tcp_id)
    print(f"[안내] 'tcp' 사이트가 없어 마지막 바디 '{body_name}'(ID: {tcp_id})를 말단 좌표로 대체합니다.")

# 결과 저장을 위한 리스트
safe_points = []
unsafe_points = []

# 각 관절의 하드웨어 리미트 범위 추출
joint_ranges = [model.jnt_range[i] for i in range(NUM_JOINTS)]

# GLFW 초기화 및 윈도우 설정
glfw.init()
window = glfw.create_window(800, 600, "JDCobot200 Safe Zone Workspace", None, None)
glfw.make_context_current(window)

# 시각화 객체 생성
scene = mj.MjvScene(model, maxgeom=5000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 카메라 초기 위치 세팅
cam.lookat[:] = [0.0, 0.0, 0.2]
cam.distance = 1.5
cam.azimuth = 135.0
cam.elevation = -25.0

print(f"\n>>> Safe Zone 탐색을 시작합니다. 총 샘플 수: {TOTAL_SAMPLES}...")

sample_count = 0

while not glfw.window_should_close(window) and sample_count < TOTAL_SAMPLES:
    # 1. 관절 범위 내 몬테카를로 무작위 샘플링
    random_angles = []
    for j in range(NUM_JOINTS):
        low, high = joint_ranges[j]
        random_angles.append(np.random.uniform(low, high))
    random_angles = np.array(random_angles)
    
    # 2. MuJoCo 가상 포워드 기구학 적용
    data.qpos[:NUM_JOINTS] = random_angles
    mj.mj_forward(model, data) 
    
    # 3. 안전한 방법으로 TCP 좌표 읽기 (IndexError 방지)
    if use_site:
        tcp_pos = data.site_xpos[tcp_id].copy()
    else:
        tcp_pos = data.xpos[tcp_id].copy()
        
    # 4. 안전성 검증 (Safe Zone & Collision Filter)
    is_safe = True
    
    # 조건 A: 바닥면 파고들기 제한 (Z축 기준)
    if tcp_pos[2] < Z_FLOOR_LIMIT:
        is_safe = False
        
    # 조건 B: 로봇 링크 간 자가 충돌 및 바닥 충돌 체크
    if data.ncon > 0:
        is_safe = False
        
    # 5. 데이터 분류 기록
    if is_safe:
        safe_points.append(tcp_pos)
    else:
        unsafe_points.append(tcp_pos)
        
    sample_count += 1
    if sample_count % 500 == 0:
        print(f"진행률: {sample_count}/{TOTAL_SAMPLES} (Safe: {len(safe_points)} | Unsafe: {len(unsafe_points)})")

    # 6. 시뮬레이션 실시간 렌더링
    viewport = glfw.get_framebuffer_size(window)
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(mj.MjrRect(0, 0, viewport[0], viewport[1]), scene, ctx)
    
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
print("\n>>> 탐색 완료. Matplotlib 3D 시각화를 시작합니다...")

# ==========================================
# 7. Safe Zone 및 Workspace 3D 시각화 (Matplotlib)
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 안전 구역 (초록색 점)
if len(safe_points) > 0:
    safe_pts = np.array(safe_points)
    ax.scatter(safe_pts[:, 0], safe_pts[:, 1], safe_pts[:, 2], 
               c='green', marker='.', s=10, alpha=0.6, label=f'Safe Zone ({len(safe_points)})')

# 위험 구역 (빨간색 점)
if len(unsafe_points) > 0:
    unsafe_pts = np.array(unsafe_points)
    ax.scatter(unsafe_pts[:, 0], unsafe_pts[:, 1], unsafe_pts[:, 2], 
               c='red', marker='.', s=5, alpha=0.1, label=f'Unsafe Zone ({len(unsafe_points)})')

ax.set_xlabel('X Coordinate (m)')
ax.set_ylabel('Y Coordinate (m)')
ax.set_zlabel('Z Coordinate (m)')
ax.set_title('JDCobot200 3D Safe Zone & Workspace')
ax.legend(loc='upper right')

# 축 비율 균등화 로직
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

output_filename = 'safe_zone_workspace.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=150)
plt.close()

print(f">>> 분석 이미지가 '{output_filename}' 파일로 저장되었습니다.")