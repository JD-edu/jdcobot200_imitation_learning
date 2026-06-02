import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# 1. 모델 로딩
xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# 조그 제어를 위한 변수 설정
active_joint_idx = 0  # 현재 제어 중인 관절 ID (0: base ~ 4: gripper)
target_qpos = np.zeros(model.nu)

joint_names = ["base", "shoulder", "elbow", "wrist", "gripper_base"]

kp, kd = 500.0, 10.0

# GLFW 키보드 콜백 함수 등록 (실시간 관절 조작용)
def key_callback(window, key, scancode, action, mods):
    global active_joint_idx, target_qpos
    if action == glfw.PRESS or action == glfw.REPEAT:
        # 숫자키 1~5로 제어할 관절 선택
        if glfw.KEY_1 <= key <= glfw.KEY_5:
            active_joint_idx = key - glfw.KEY_1
            print(f"\n>> 현재 선택된 관절: [{joint_names[active_joint_idx]}]")
        
        # 방향키 위/아래로 각도 미세 조절 (약 3도씩 조절)
        elif key == glfw.KEY_UP:
            target_qpos[active_joint_idx] += 0.05
        elif key == glfw.KEY_DOWN:
            target_qpos[active_joint_idx] -= 0.05
            
        # 현재 모든 관절의 라디안 각도를 실시간 모니터링 출력
        print(f"현재 각도(Rad) -> Base: {data.qpos[0]:.3f} | Shoulder: {data.qpos[1]:.3f} | Elbow: {data.qpos[2]:.3f} | Wrist: {data.qpos[3]:.3f} | Gripper: {data.qpos[4]:.3f}", end='\r')

# GLFW 초기화
glfw.init()
window = glfw.create_window(1280, 720, "JdCobot200 Joint Limit Finder", None, None)
glfw.make_context_current(window)
glfw.set_key_callback(window, key_callback)

scene = mj.MjvScene(model, maxgeom=1000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cam.lookat[:] = [0.0, 0.0, 0.1]
cam.distance = 1.2
cam.azimuth = 135.0
cam.elevation = -20.0

print("="*60)
print(" 사용 방법:")
print(" 1. 숫자키 [1, 2, 3, 4, 5]를 눌러 움직일 관절을 선택합니다.")
print(" 2. 방향키 [▲ / ▼] 를 꾹 누르거나 연타하여 관절을 회전시킵니다.")
print(" 3. 링크끼리 닿기 직전 혹은 바닥에 닿기 직전의 터미널 각도를 기록하세요.")
print("="*60)

while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev) < (1.0/60.0):
        # 파이썬 레벨 PD 제어
        position_error = target_qpos - data.qpos[:model.nu]
        velocity_error = 0 - data.qvel[:model.nu]
        data.ctrl[:model.nu] = (kp * position_error) + (kd * velocity_error)
        mj.mj_step(model, data)

    # 렌더링
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, ctx)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()