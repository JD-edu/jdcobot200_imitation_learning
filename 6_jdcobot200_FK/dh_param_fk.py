import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# --- 1. 업로드된 파일(dh_param_fk.py)의 정석 DH 수식 적용 ---
def get_dh_matrix(a, alpha, d, theta):
    """
    표준 DH 파라미터 4가지 변수를 받아 4x4 동차 변환 행렬을 반환하는 함수
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [ 0,       sa,       ca,      d],
        [ 0,        0,        0,      1]
    ])
    return T

def jdcobot200_forward_kinematics(joint_angles):
    """
    jdcobot200의 5개 관절 각도(라디안)를 받아 말단(End-Effector)의 위치와 자세를 계산
    """
    # [참고 파일 내 정밀 정렬된 Standard DH 테이블]
    dh_table = [
        [0.0,      np.pi/2,  0.0537,  0.0],       # 1. Base
        [0.1352,   0.0,      0.06146, np.pi/2],   # 2. Shoulder
        [0.1352,   0.0,      0.0,     0.0],       # 3. Elbow 
        [0.0,      -np.pi/2, 0.0,     -np.pi/2],  # 4. Wrist Pitch
        [0.0,      0.0,      0.0575,  0.0]        # 5. Wrist Roll -> EE Tip
    ]
    
    T_total = np.eye(4)
    
    for i, (a, alpha, d, theta_offset) in enumerate(dh_table):
        # 주관절 5개 축의 각도에 기하학적 정렬 오프셋 적용
        current_theta = joint_angles[i] + theta_offset
        T_i = get_dh_matrix(a, alpha, d, current_theta)
        T_total = np.dot(T_total, T_i)
        
    return T_total

# --- 2. MuJoCo 모델 및 환경 설정 ---
xml_path = os.path.join(os.path.dirname(__file__), "jdcobot200.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# --- 3. 실험용 2지점 목표 포즈 및 제어 파라미터 ---
# jdcobot200의 5자유도 주관절에 맞추어 각도 지정 (그리퍼 제외 5축 매핑)
pose_A = np.radians([0.0, 0.0, 0.0, 0.0, 0.0])       # 지점 A (일직선 홈 포즈)
pose_B = np.radians([30.0, 45.0, 45.0, -30.0, 0.0])  # 지점 B (임의의 구동 포즈)

kp = 400.0            # 위치 제어 비례 이득
kd = 15.0             # 댐핑 이득
switch_interval = 2.5 # 2.5초마다 지점 A <-> B 전환

# GLFW 윈도우 초기화
if not glfw.init():
    raise Exception("GLFW 초기화 실패")

window = glfw.create_window(800, 600, "JDCobot200 DH-FK Control Experiment", None, None)
if not window:
    glfw.terminate()
    raise Exception("윈도우 생성 실패")

glfw.make_context_current(window)

# 시각화 데이터 구조 설정
scene = mj.MjvScene(model, maxgeom=1000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 카메라 각도 조절
cam.lookat[:] = [0.0, 0.0, 0.2]
cam.distance = 1.2
cam.azimuth = 135.0
cam.elevation = -25.0

# 제어 변수 초기화
last_switch_time = 0.0
current_target = pose_A

print(">> DH 파라미터 기반 FK 실험용 시뮬레이션을 시작합니다.")

# --- 4. 시뮬레이션 메인 루프 ---
while not glfw.window_should_close(window):
    time_now = data.time
    
    # 설정 시간마다 목표 지점(A or B) 교체
    if time_now - last_switch_time > switch_interval:
        if np.array_equal(current_target, pose_A):
            current_target = pose_B
            print("\n[이동] >>> 목표 지점 B로 구동 시작")
        else:
            current_target = pose_A
            print("\n[이동] >>> 목표 지점 A로 구동 시작")
        last_switch_time = time_now

    # 서보 모터 모사를 위한 PD 제어 입력 인가 (5개 주관절 구동)
    for i in range(5):
        data.ctrl[i] = kp * (current_target[i] - data.qpos[i]) - kd * data.qvel[i]
        
    # 물리 엔진 한 스텝 계산
    mj.mj_step(model, data)
    
    # 실시간 직접 구현한 DH FK 결과 터미널 출력 (일정 주기로 스크롤 다운 조절)
    if int(time_now * 100) % 25 == 0:
        # 현재 시뮬레이터 상의 실제 로봇 관절각 추출
        current_angles = data.qpos[:5]
        
        # 참고 문서 수식 기반으로 FK XYZ 위치 연산
        T_total = jdcobot200_forward_kinematics(current_angles)
        dh_xyz = T_total[:3, 3]
        
        print(f"[DH FK 연산 결과] XYZ: {np.round(dh_xyz, 4)} m | 현재 관절각(deg): {np.round(np.degrees(current_angles), 1)}")

    # 그래픽 화면 업데이트
    viewport = glfw.get_framebuffer_size(window)
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(mj.MjrRect(0, 0, viewport[0], viewport[1]), scene, ctx)
    
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()