import mujoco as mj
from mujoco.glfw import glfw
import os

# 1. 모델 및 데이터 로딩 (가장 기본적이고 필수적인 부분)
xml_path = os.path.join(os.path.dirname(__file__), "jdcobot200.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# 2. GLFW 윈도우 생성 및 설정
glfw.init()
window = glfw.create_window(800, 600, "Simple Model Viewer", None, None)
glfw.make_context_current(window)

# 3. 시각화 및 카메라 객체 생성
scene = mj.MjvScene(model, maxgeom=1000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 초기 카메라 위치 설정
cam.lookat[:] = [0.0, 0.0, 0.1]
cam.distance = 1.2
cam.azimuth = 135.0
cam.elevation = -20.0

# 4. 메인 시뮬레이션 및 렌더링 루프
while not glfw.window_should_close(window):
    time_prev = data.time

    # 1/60초만큼 물리 시뮬레이션 진행 (단순 물리 스텝만 수행)
    while (data.time - time_prev) < (1.0 / 60.0):
        mj.mj_step(model, data)

    # 화면에 로봇 그리기 (렌더링)
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()

# 종료 처리
glfw.terminate()