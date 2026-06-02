import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# 1. 모델 및 데이터 로딩 (기존 사양 완벽 동기화)
xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# --- [핵심 수정] 명시적 액추에이터 ID 검색 장치 ---
# 파이썬 배열 번호(0, 1, 2) 대신 XML에 적힌 고유 name 문자열로 하드웨어 포트를 추적합니다.
actuator_base_id     = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "velocity_base")
actuator_shoulder_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "torque_shoulder")
actuator_elbow_id    = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "torque_elbow")
actuator_wrist_id    = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "torque_wrist")
actuator_gripper_id  = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "torque_gripper_base")

# GLFW 초기화 및 가상 윈도우 세팅
glfw.init()
window = glfw.create_window(1280, 720, "jdcobot200 Smooth Velocity Control Mode", None, None)
glfw.make_context_current(window)

scene = mj.MjvScene(model, maxgeom=1000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 초기 카메라 시점 세팅
cam.lookat[:] = [0.1, -0.1, -0.05] 
cam.distance = 1.0             
cam.azimuth = 135.0             
cam.elevation = -25.0          

print("="*60)
print(" MuJoCo 시뮬레이션 가동 시작")
print(" - 1번 베이스 축: 사인파 기반 가감속 속도 제어 (부드러운 좌우 왕복)")
print(" - 2~5번 관절 축: 원점 포즈(0.0 rad) 내장 위치 홀딩 제어")
print("="*60)

# 시뮬레이션 메인 루프
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev) < (1.0 / 60.0):
        
        # [안전 메커니즘] 사인파(Sine Wave)를 이용한 부드러운 가감속 속도 생성
        # 3초 주기로 최대 1.0 rad/s ~ -1.0 rad/s 사이를 부드럽게 감속하며 왕복합니다.
        # 이 방식을 쓰면 속도 제어 도중 관절 리미트 한계점에 도달하기 전에 스스로 방향을 바꿉니다.
        target_velocity = 1.0 * np.sin(2 * np.pi * data.time / 3.0)
        
        # 1. 베이스 축 (속도 제어): 목표 속도 대입
        data.ctrl[actuator_base_id] = target_velocity

        # 2. 나머지 관절 (위치 제어): 원점 자리를 끈적하게 홀딩하도록 0.0 인입
        data.ctrl[actuator_shoulder_id] = 0.0
        data.ctrl[actuator_elbow_id]    = 0.0
        data.ctrl[actuator_wrist_id]    = 0.0
        data.ctrl[actuator_gripper_id]  = 0.0

        # 물리 엔진 1스텝 전진 연산
        mj.mj_step(model, data)

    # 1초 간격으로 현재 타겟 속도와 실제 회전 속도 모니터링 출력
    if int(data.time * 10) % 10 == 0:
        print(f"Time: {data.time:.1f}s | 명령 속도: {target_velocity:+.2f} rad/s | 실제 베이스 속도: {data.qvel[0]:+.2f} rad/s")

    # 가상 시각화 렌더링 스텝
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()