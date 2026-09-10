# Astra 카메라 ↔ JDCobot200 IK 좌표 캘리브레이션

현재 베이스에 고정한 ArUco 하나로 **고정 카메라 좌표에서 로봇 베이스 좌표로의
6-DoF 변환**을 구할 수 있다. 마커 한 변 길이와, 마커 중심/축이 로봇 IK 원점에
대해 어디에 있는지는 자로 실측해야 한다. 영상만으로 이 오프셋은 결정할 수 없다.

## 좌표 정의

- physical baseframe: `so101_new_calib.xml`의 `baseframe` 사이트
- IK/world: solver가 목표값으로 받는 `scene.xml` world 좌표, 단위 m. 현재 모델에서
  physical baseframe 원점은 IK/world `(0.1, 0, 0.2) m`에 있다.
- camera: OpenCV 좌표 (`+X` 영상 오른쪽, `+Y` 아래, `+Z` 카메라 전방)
- marker: 인쇄면을 정면으로 보았을 때 `+X` 오른쪽, `+Y` 위쪽. `+Z`는 마커 면의
  법선이며 카메라를 향한다.

마커 중심의 **physical baseframe 기준** 좌표(mm)와 marker 축의 baseframe 기준
roll/pitch/yaw(deg)를 잰다.
회전 적용 순서는 `Rz(yaw) Ry(pitch) Rx(roll)`이다. 예를 들어 마커를 수평으로
놓았다면 marker `+Z`가 위를 향하도록 RPY를 정해야 하며, 인쇄물의 위쪽 방향에
따라 yaw가 달라진다.

## 실행

아래 숫자는 예시이므로 실제 측정값으로 바꾼다.

```bash
python3 astra_aruco_calibration.py \
  --marker-size-mm 50 \
  --marker-id 0 \
  --marker-in-base-mm 120 0 0 \
  --marker-rpy-deg 180 0 0 \
  --preview
```

기본 `--ik-origin-from-base-mm 100 0 200`은 현재 XML에서 자동 변환에 반영된다.
XML의 base body 위치를 바꾸었다면 이 옵션도 같은 값으로 바꾼다.

마커가 검출되지 않으면 `--snapshot aruco_check.png`로 마지막 RGB 영상을 저장해
시야와 인쇄 상태를 확인한다. 기본 `AUTO`는 흔한 사전들을 검색하며, 생성 사전을
안다면 예를 들어 `--dictionary DICT_5X5_100`처럼 지정하는 편이 빠르다.

현재 촬영된 마커는 `DICT_4X4_1000`, ID `871`로 확인되었으므로 다음 옵션을 직접
지정하면 가장 안정적이다.

```bash
--dictionary DICT_4X4_1000 --marker-id 871
```

40개 유효 프레임을 모으면 `camera_to_robot_calibration.json`이 생성된다. 출력의
재투영 오차는 가급적 1 px 이하, translation std는 수 mm 이하가 좋다. Astra가
보고하는 FOV로 내부 파라미터를 근사하므로 정밀 작업에는 체커보드로 구한 RGB
`fx fy cx cy`를 `--fx --fy --cx --cy`로 넣는다.

카메라의 3D 점(단위 m)을 IK 좌표로 바꾸는 예:

```bash
python3 camera_robot_transform.py 0.03 -0.02 0.55
```

코드에서는 `camera_robot_transform.camera_to_base(points, matrix)`를 사용한다.
깊이 픽셀은 먼저 `depth_pixel_to_camera(u, v, depth_mm, camera_matrix)`로 3D화한
뒤 변환한다. 단, Astra RGB와 depth 픽셀은 서로 정렬되지 않았으므로 RGB에서 찾은
픽셀에 depth 값을 그대로 대입하면 안 된다. OpenNI depth-to-color registration이나
별도의 RGB-depth 외부 파라미터가 필요하다.

그리퍼 마커는 이 고정 카메라→베이스 변환에는 필요 없다. 이후 실제 관절 FK와
그리퍼 끝점(TCP)의 오프셋을 검증하거나 hand-eye 보정을 할 때 추가하면 된다.

## 마커 상공 100 mm 이동 검증

먼저 카메라 측정만 수행한다. 이 모드는 로봇에 연결하지 않는다.

```bash
python3 move_above_aruco.py
```

출력 좌표와 `aruco_target_check.png`를 확인한 다음 MuJoCo에서 IK와 safe-zone을
검사한다.

```bash
python3 move_above_aruco.py --mode simulate
```

마지막으로 주변을 비우고 비상정지가 가능한 상태에서 실제 이동을 선택한다.
프로그램이 현재 엔코더, READY 경로, 관절 제한, 충돌, 바닥 여유와 특이점을 먼저
검사한다. 검사 후에도 터미널에 정확히 `MOVE`를 입력해야 토크가 켜진다.

```bash
python3 move_above_aruco.py --mode move
```
