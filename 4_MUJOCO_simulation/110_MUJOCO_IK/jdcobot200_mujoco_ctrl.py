import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# ===== IK 설정 =====
ee_body_name = "gripper_assembly"
ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body_name)

joint_names = ["base", "shoulder", "elbow", "wrist", "gripper_base"]
joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
qpos_ids = [model.jnt_qposadr[jid] for jid in joint_ids]
dof_ids = [model.jnt_dofadr[jid] for jid in joint_ids]

# xyz 목표점 2개
target_A = np.array([0.18,  0.08, 0.10])
target_B = np.array([0.18, -0.08, 0.16])

switch_interval = 2.0

ik_iters = 15
ik_alpha = 0.6
ik_damping = 0.03
ik_tol = 1e-3

# actuator는 position actuator이므로 ctrl에는 목표 joint angle을 넣음
q_des = data.qpos[qpos_ids].copy()

# ===== GLFW =====
glfw.init()
window = glfw.create_window(800, 600, "MuJoCo Numerical IK XYZ Control", None, None)
glfw.make_context_current(window)

scene = mj.MjvScene(model, maxgeom=1000)
cam = mj.MjvCamera()
ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cam.lookat[:] = [0.0, 0.0, 0.08]
cam.distance = 0.8
cam.azimuth = 45.0
cam.elevation = -25.0


def numerical_ik(target_pos):
    global q_des

    q = data.qpos.copy()
    q[qpos_ids] = q_des

    for _ in range(ik_iters):
        data.qpos[:] = q
        data.qvel[:] = 0
        mj.mj_forward(model, data)

        ee_pos = data.xpos[ee_id].copy()
        err = target_pos - ee_pos

        if np.linalg.norm(err) < ik_tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jacp, jacr, ee_id)

        J = jacp[:, dof_ids]

        # Damped Least Squares:
        # dq = J.T @ inv(J @ J.T + lambda^2 I) @ error
        A = J @ J.T + (ik_damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)

        q_new = q[qpos_ids] + ik_alpha * dq

        # joint limit 적용
        for i, jid in enumerate(joint_ids):
            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
                q_new[i] = np.clip(q_new[i], low, high)

        q[qpos_ids] = q_new

    q_des = q[qpos_ids].copy()
    return q_des


while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev) < (1.0 / 60.0):
        if int(data.time // switch_interval) % 2 == 0:
            target_pos = target_A
        else:
            target_pos = target_B

        target_q = numerical_ik(target_pos)

        # position actuator 제어
        data.ctrl[:len(target_q)] = target_q

        mj.mj_step(model, data)

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    mj.mjv_updateScene(
        model, data, mj.MjvOption(), None, cam,
        mj.mjtCatBit.mjCAT_ALL, scene
    )
    mj.mjr_render(viewport, scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()