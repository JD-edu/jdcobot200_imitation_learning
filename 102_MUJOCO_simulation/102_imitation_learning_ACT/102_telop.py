import time
import json
import threading
import os
import mujoco as mj
from mujoco.glfw import glfw
from motor_control import MiniFeetechDriver

# --- 하드웨어 및 파일 경로 구성 설정 ---
CALIB_FILE = "./calibration.json"
XML_PATH = "scene.xml"  # 질량 버그가 수정된 jdcobot200 xml 파일 경로
PORT = "/dev/ttyUSB0"
BAUDRATE = 1000000

# 글로벌 스레드 공유 데이터 메모리 구조 (스레드 세이프 교환용)
shared_qpos = {}
running = True

def hardware_leader_reader_thread():
    """ 실제 리더 로봇(Leader)의 모터 엔코더 값을 읽어 
        논리적 2047 원점 기준으로 보정하고 라디안으로 맵핑하는 백그라운드 스레드 """
    global shared_qpos, running
    
    # 1. 수동 작성된 칼리브레이션 프로파일 존재 여부 예외 처리
    if not os.path.exists(CALIB_FILE):
        print(f"[오류] {CALIB_FILE} 파일이 존재하지 않습니다. 수동 작성을 먼저 완료하세요.")
        running = False
        return
        
    with open(CALIB_FILE, "r", encoding="utf-8") as f:
        calib_map = json.load(f)
    print("[OK] 수동 텔레오퍼레이션 매핑 데이터 로드 완료.")

    # 2. Feetech 모터 드라이버 연결 및 토크 오프 (사람이 손으로 잡고 움직이는 조작 모드)
    try:
        driver = MiniFeetechDriver(port=PORT, baudrate=BAUDRATE)
        for joint_name, config in calib_map.items():
            # 사용자가 손으로 부드럽게 티칭/운용할 수 있도록 토크를 완전 해제합니다.
            driver.set_torque(config["id"], False) 
    except Exception as e:
        print(f"[하드웨어 에러] 모터 통신 인터페이스 연결 실패: {e}")
        running = False
        return

    print("\n" + "="*60)
    print(" >>> 텔레오퍼레이션 시스템 가동! 실제 로봇을 손으로 움직여 보세요.")
    print("="*60 + "\n")
    
    while running:
        temp_qpos = {}
        for joint_name, config in calib_map.items():
            m_id = config["id"]
            
            # 7번 샘플링 필터링 함수가 있다면 사용하여 엔코더 센서 노이즈와 손떨림을 제어합니다.
            try:
                if hasattr(driver, 'get_position_filtered'):
                    raw_tick = driver.get_position_filtered(m_id, samples=7)
                else:
                    raw_tick = driver.get_position(m_id)
            except:
                continue

            if raw_tick is None:
                continue

            # [핵심 수식] 오프셋을 반영하여 물리 틱을 논리적인 2047 중심 틱으로 변환
            # (수동 JSON에서 homing_offset을 0으로 둔 경우 raw_tick이 그대로 사용됩니다)
            calibrated_tick = raw_tick - config["homing_offset"]
            
            # 하드웨어의 최대/최소 한계 영역을 벗어나지 않도록 클리핑 안전장치 작동
            calibrated_tick = max(config["hw_range_min"], min(config["hw_range_max"], calibrated_tick))
            
            # 논리적 틱 범위(0 ~ 4095)를 MuJoCo XML 라디안 범위로 선형 보간 매핑 (Linear Interpolation)
            tick_range = config["hw_range_max"] - config["hw_range_min"] + 1e-6
            tick_ratio = (calibrated_tick - config["hw_range_min"]) / tick_range
            
            sim_rad = config["sim_range_min"] + tick_ratio * (config["sim_range_max"] - config["sim_range_min"])
            
            temp_qpos[joint_name] = sim_rad
            
        # 연산된 라디안 상태 배열을 메인 시뮬레이터 루프로 안전하게 전달
        shared_qpos = temp_qpos
        time.sleep(0.015)  # 66Hz 주기로 하드웨어 고속 패킷 구독

def main():
    global running
    
    # 1. 질량 버그가 전면 보정된 MuJoCo 모델 및 데이터 구조체 로드
    try:
        model = mj.MjModel.from_xml_path(XML_PATH)
        data = mj.MjData(model)
    except Exception as e:
        print(f"[시뮬레션 에러] XML 모델을 읽을 수 없습니다: {e}")
        return
    
    # 2. 마스터 로봇 각도 수집을 위한 백그라운드 스레드 비동기 분리 구동
    reader_thread = threading.Thread(target=hardware_leader_reader_thread, daemon=True)
    reader_thread.start()

    # 3. GLFW 시각화 윈도우 인터페이스 초기화 및 디스플레이 셋팅
    if not glfw.init():
        return
        
    window = glfw.create_window(1000, 800, "jdcobot200 - Real-to-Sim Teleoperation Follower", None, None)
    if not window:
        glfw.terminate()
        return
        
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # V-Sync 동기화 활성화

    # MuJoCo 렌더링용 내부 시각화 컨텍스트 구성
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=1000)
    ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    # 로봇이 정중앙에 시각적으로 잘 들어오도록 뷰포트 카메라 초기 시점 조정
    cam.lookat[:] = [0.12, -0.09, 0.15]
    cam.distance = 1.3
    cam.azimuth = 145
    cam.elevation = -20

    print("[OK] MuJoCo 시뮬레이터 실시간 관측 창 셋업 완료.")

    # 4. 실시간 마스터-슬레이브 동기화 기하 렌더링 루프 (Kinematic Injection)
    while not glfw.window_should_close(window):
        if not running:
            break
            
        # 비동기 스레드로부터 정제 완료된 최신 라디안 딕셔너리 안전 복사
        current_shared_qpos = shared_qpos.copy()
        
        # 전송받은 관절명 사양을 순회하며 MuJoCo 시뮬레이션의 관절 주소(qpos)에 다이렉트 주입
        for joint_name, rad_val in current_shared_qpos.items():
            try:
                # 조인트 문자열 이름을 기반으로 MuJoCo 고유 인덱스 ID 조회
                joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id == -1:
                    continue
                
                # 해당 조인트가 매핑되어 있는 qpos의 메모리 시작 주소 획득
                qpos_adr = model.jnt_qposadr[joint_id]
                
                # 강제 기하학적 매핑 제어 적용 (동역학 폭주 원천 차단)
                data.qpos[qpos_adr] = rad_val
            except:
                pass

        # 물리적 기하 변환 및 위치 전방향 상태 업데이트 보장
        mj.mj_forward(model, data)
        
        # 뷰포트 영역 계산 및 동적 그래픽스 렌더링
        viewport = glfw.get_framebuffer_size(window)
        mj_viewport = mj.MjrRect(0, 0, viewport[0], viewport[1])
        
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(mj_viewport, scene, ctx)
        
        glfw.swap_buffers(window)
        glfw.poll_events()

    # 안전 프로그램 종료 루틴
    running = False
    reader_thread.join()
    glfw.terminate()
    print("[시스템 안내] jdcobot200 텔레오퍼레이션 연동 프로그램이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()