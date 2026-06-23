import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# 1. 모델 로딩
# JDCobot200 xml 파일명에 맞게 조정하세요. (예: jdcobot200.xml)
xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# --- 말단 그리퍼(TCP) Body ID 가져오기 ---
# 'gripper_base' 또는 실제 XML 상의 최말단 body/site 이름으로 매칭하세요.
try:
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "gripper_base")
except ValueError:
    # 이름이 다를 경우 0번(기본값) 혹은 마지막 바디 ID 사용
    ee_id = model.nbody - 1 

# --- 제어 설정 ---
# JDCobot200의 5자유도 주관절 + 그리퍼 구동축에 맞춤 (6개)
pose_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
pose_B = np.array([0.5, 0.5, -0.3, 0.2, -0.5, 0.0])

kp = 500.0  # 위치 이득
kd = 10.0   # 속도 이득 (댐핑)
switch_interval = 2.0  # 2초마다 포즈 변경

# GLFW 초기화 및 윈도우 설정
if not glfw.init():
    raise Exception("GLFW 초기화 실패")

window = glfw.create_window(800, 600, "JDCobot200 MuJoCo FK Control", None, None)
if not window:
    glfw.terminate()
    raise Exception("윈도우 생성 실패")

glfw.make_context_current(window)

# 시각화 객체 생성
scene = mj.MjvScene(model, maxgeom=1000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 초기 카메라 위치 설정
cam.lookat[:] = [0.0, 0.0, 0.2]
cam.distance = 1.0
cam.azimuth = 135.0
cam.elevation = -20.0

# --- [핵심] MuJoCo 내장 FK 함수 정의 ---
def get_mujoco_fk(joint_angles):
    """
    MuJoCo의 가상 데이터 객체를 이용해 
    실제 시뮬레이션을 방해하지 않고 특정 각도에서의 TCP 위치를 미리 계산하는 FK 함수
    """
    # 임시 데이터 객체 생성 및 관절 각도 대입
    temp_data = mj.MjData(model)
    temp_data.qpos[:len(joint_angles)] = joint_angles
    
    # MuJoCo 정방향 기구학(FK) 엔진 강제 업데이트
    mj.mj_forward(model, temp_data)
    
    # 말단 TCP(ee_id)의 X, Y, Z 위치 및 3x3 회전행렬 반환
    tcp_position = temp_data.xpos[ee_id].copy()
    tcp_orientation = temp_data.xmat[ee_id].reshape(3, 3).copy()
    
    return tcp_position, tcp_orientation

# 루프 및 시간 초기화
last_switch_time = 0.0
current_target = pose_A

while not glfw.window_should_close(window):
    time_now = data.time
    
    # 2초마다 목표 포즈 교체
    if time_now - last_switch_time > switch_interval:
        if np.array_equal(current_target, pose_A):
            current_target = pose_B
            print("\n>> 목표 포즈 B로 전환 변경")
        else:
            current_target = pose_A
            print("\n>> 목표 포즈 A로 전환 변경")
        last_switch_time = time_now
    
    # --- PD 제어기 구동 (로봇 움직이기) ---
    for i in range(len(current_target)):
        # 각 관절의 제어 입력(액추에이터)에 토크 인가
        data.ctrl[i] = kp * (current_target[i] - data.qpos[i]) - kd * data.qvel[i]
        
    # 물리 시뮬레이션 한 스텝 전진 (내부적으로 mj_forward 자동 호출됨)
    mj.mj_step(model, data)
    
    # --- 실시간 MuJoCo FK 결과 출력 (10스텝마다 한 번씩 출력) ---
    if int(data.time * 100) % 20 == 0:
        # 현재 실제 관절 각도 기반으로 FK 계산
        current_angles = data.qpos[:6]
        tcp_xyz, tcp_rot = get_mujoco_fk(current_angles)
        
        print(f"[실시간 MuJoCo FK] 현재 관절각: {np.round(current_angles, 3)}")
        print(f"               Calculated TCP XYZ: {np.round(tcp_xyz, 4)} (단위: m)")

    # 화면 렌더링 업데이트
    viewport = glfw.get_framebuffer_size(window)
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(mj.MjrRect(0, 0, viewport[0], viewport[1]), scene, ctx)
    
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()