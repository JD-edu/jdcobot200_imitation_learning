import time
import serial
from motor_control import MiniFeetechDriver  # 사용자 환경에 맞는 파일명/클래스명

def load_gripper_offset(file_path="joint_limits.txt"):
    """별도의 그리퍼 오프셋 파일에서 오프셋 값을 읽어옵니다."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    motor_id, offset_val = line.split("=")
                    if int(motor_id) == 7:
                        return int(offset_val)
    except FileNotFoundError:
        print("⚠️ [경고] gripper_offset.txt 파일이 없어 오프셋을 0으로 간주합니다.")
    return 0

def measure_gripper_limits():
    # ==========================================
    # [설정 영역] 그리퍼 ID 및 포트 설정
    # ==========================================
    PORT = "/dev/ttyACM0"           # 시리얼 포트 경로
    BAUDRATE = 1000000             # 통신 속도 (1Mbps)
    GRIPPER_MOTOR_ID = 6           # 그리퍼 서보모터 ID
    LIMIT_FILE_NAME = "joint_limits.txt"  # 누적 저장할 리미트 파일명
    SAFE_MARGIN = 50               # 기어 보호용 안전 마진 스텝

    # 1. 드라이버 초기화 및 오프셋 로드
    driver = MiniFeetechDriver(port=PORT, baudrate=BAUDRATE)
    gripper_offset = load_gripper_offset()
    
    print("=" * 60)
    print(f"★ [ID: {GRIPPER_MOTOR_ID}] 그리퍼 가동범위 측정 프로그램 ★")
    print(f"   로드된 그리퍼 오프셋: {gripper_offset}")
    print("=" * 60)
    
    # 2. 그리퍼 모터만 콕 집어서 토크 해제 (Release)
    driver.set_torque(GRIPPER_MOTOR_ID, False)
    print("\n🔓 그리퍼 토크 해제 완료! 손으로 그리퍼를 작동해 보세요.")
    print("👉 완전히 벌렸을 때와 완전히 다물었을 때까지 천천히 움직여 줍니다.")
    print("👉 측정을 완료했다면 [Ctrl + C]를 눌러 저장하고 종료하세요.\n")
    print("-" * 60)

    # 측정을 위한 초기값 설정
    min_raw_pos = 4096
    max_raw_pos = 0

    try:
        while True:
            # 현재 위치 엔코더 값 읽기 (레지스터 56번)
            raw_pos = driver.read_u16(GRIPPER_MOTOR_ID, 56)
            
            if raw_pos == 0 or raw_pos > 4095:
                time.sleep(0.05)
                continue
            
            # 실시간 최소/최대 물리 위치 갱신
            if raw_pos < min_raw_pos:
                min_raw_pos = raw_pos
            if raw_pos > max_raw_pos:
                max_raw_pos = raw_pos

            # 오프셋이 적용된 실시간 가상 좌표 계산
            current_mapped_pos = raw_pos - gripper_offset
            
            print(f"\r[측정중] 실제값: {raw_pos:4d} | 오프셋반영값: {current_mapped_pos:4d} | "
                  f"물리최소: {min_raw_pos:4d} 물리최대: {max_raw_pos:4d}", end="", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("측정이 종료되었습니다. 안전 마진 및 오프셋을 적용합니다.")
        
        # 3. 측정 종료 후 즉시 그리퍼 토크를 다시 켜서 고정 (Freeze)
        driver.set_torque(GRIPPER_MOTOR_ID, True)
        
        if max_raw_pos == 0 or min_raw_pos == 4096:
            print("측정된 포지션 데이터가 유효하지 않아 저장하지 않고 종료합니다.")
            return

        # 4. 오프셋이 반영된 가상 원점 기준의 리미트 좌표 계산
        mapped_min = min_raw_pos - gripper_offset
        mapped_max = max_raw_pos - gripper_offset

        # 5. 하드웨어 충격 방지용 안전 마진(±50) 적용
        safe_min = mapped_min + SAFE_MARGIN
        safe_max = mapped_max - SAFE_MARGIN

        print(f"\n✨ [최종 계산 결과]")
        print(f"   · 오프셋 반영 리미트 (마진 전) : Mapped Min={mapped_min}, Mapped Max={mapped_max}")
        print(f"   · 안전 마진 적용 소프트웨어 제한: Safe Min={safe_min}, Safe Max={safe_max}")
        
        # 6. 기존 joint_limits.txt 파일의 맨 아래에 누적(Append) 저장
        try:
            with open(LIMIT_FILE_NAME, "a", encoding="utf-8") as f:
                f.write(f"{GRIPPER_MOTOR_ID}={safe_min},{safe_max}\n")
            print(f"\n💾 [{LIMIT_FILE_NAME}] 파일 맨 아래에 그리퍼 리미트가 누적 저장되었습니다!")
        except Exception as e:
            print(f"파일 저장 실패: {e}")
            
        print("🔒 그리퍼 토크 잠금 고정 및 프로그램을 안전하게 종료합니다.")

if __name__ == "__main__":
    measure_gripper_limits()