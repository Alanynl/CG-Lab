import taichi as ti
import numpy as np

# 初始化Taichi（GPU加速）
ti.init(arch=ti.gpu, default_fp=ti.f32, debug=False)

# 全局参数设置
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 20  # 布料网格尺寸
NUM_PARTICLES = GRID_SIZE * GRID_SIZE
DT = 5e-4  # 时间步长
MASS = 1.0  # 质点质量
KS = 10000.0  # 弹簧劲度系数
KD = 1.0  # 阻尼系数
GRAVITY = ti.Vector([0.0, -9.8, 0.0])  # 重力加速度
MAX_VELOCITY = 50.0  # 速度钳制上限
FIXED_PARTICLES = ti.field(dtype=int, shape=NUM_PARTICLES)  # 固定质点标记
IMPLICIT_ITER = 3  # 隐式欧拉迭代次数

# 物理场定义
position = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
velocity = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
force = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)

# 隐式欧拉专用缓存场
x_next = ti.Vector.field(3, ti.f32, NUM_PARTICLES)
v_next = ti.Vector.field(3, ti.f32, NUM_PARTICLES)
f_next = ti.Vector.field(3, ti.f32, NUM_PARTICLES)

# 弹簧数据
max_springs = GRID_SIZE * GRID_SIZE * 4
num_springs = ti.field(dtype=int, shape=())
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_rest_length = ti.field(dtype=ti.f32, shape=max_springs)
# 弹簧渲染索引
spring_line_indices = ti.field(dtype=int, shape=max_springs * 2)

# 模拟控制参数
current_method = ti.field(dtype=int, shape=())  # 0:显式 1:半隐式 2:隐式
paused = ti.field(dtype=int, shape=())

# 初始化
@ti.kernel
def init_positions():
    """初始化质点位置与固定点"""
    for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
        idx = i * GRID_SIZE + j
        # 布料初始位置
        position[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        velocity[idx] = ti.Vector([0.0, 0.0, 0.0])
        force[idx] = ti.Vector([0.0, 0.0, 0.0])
        # 固定顶部两个角点
        FIXED_PARTICLES[idx] = 1 if (j == 0 and (i == 0 or i == GRID_SIZE - 1)) else 0

@ti.kernel
def init_springs():
    """初始化结构弹簧"""
    num_springs[None] = 0
    for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
        idx = i * GRID_SIZE + j
        # 水平弹簧
        if i < GRID_SIZE - 1:
            right_idx = (i + 1) * GRID_SIZE + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, right_idx])
            spring_rest_length[c] = (position[idx] - position[right_idx]).norm()
        # 垂直弹簧
        if j < GRID_SIZE - 1:
            down_idx = i * GRID_SIZE + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, down_idx])
            spring_rest_length[c] = (position[idx] - position[down_idx]).norm()

@ti.kernel
def init_spring_lines():
    """初始化弹簧渲染线索引"""
    for i in range(num_springs[None]):
        spring_line_indices[i * 2] = spring_pairs[i][0]
        spring_line_indices[i * 2 + 1] = spring_pairs[i][1]

def reset_simulation():
    """重置模拟"""
    init_positions()
    init_springs()
    init_spring_lines()
    current_method[None] = 1
    paused[None] = 0
    print("Simulation reset!")

# 力计算与防爆
@ti.func
def compute_forces(pos: ti.template(), vel: ti.template(), f: ti.template()):
    """计算所有受力：重力+阻尼+弹簧力"""
    # 清空受力 + 重力+阻尼
    for i in range(NUM_PARTICLES):
        f[i] = GRAVITY * MASS - KD * vel[i]
    # 弹簧力
    for i in range(num_springs[None]):
        a, b = spring_pairs[i][0], spring_pairs[i][1]
        dx = pos[a] - pos[b]
        dist = dx.norm()
        if dist > 1e-6:
            spring_force = -KS * (dist - spring_rest_length[i]) * (dx / dist)
            ti.atomic_add(f[a], spring_force)
            ti.atomic_add(f[b], -spring_force)

@ti.func
def clamp_vel(vel: ti.template(), idx: int):
    """速度钳制，防爆"""
    speed = vel[idx].norm()
    if speed > MAX_VELOCITY:
        vel[idx] = vel[idx] / speed * MAX_VELOCITY

# 三种积分器
@ti.kernel
def step_explicit():
    """显式欧拉"""
    compute_forces(position, velocity, force)
    for i in range(NUM_PARTICLES):
        if FIXED_PARTICLES[i] == 0:
            position[i] += velocity[i] * DT
            velocity[i] += force[i] / MASS * DT
            clamp_vel(velocity, i)

@ti.kernel
def step_semi_implicit():
    """半隐式欧拉（默认）"""
    compute_forces(position, velocity, force)
    for i in range(NUM_PARTICLES):
        if FIXED_PARTICLES[i] == 0:
            velocity[i] += force[i] / MASS * DT
            clamp_vel(velocity, i)
            position[i] += velocity[i] * DT

@ti.kernel
def step_implicit():
    """隐式欧拉（定点迭代）"""
    # 初始化预测值
    for i in range(NUM_PARTICLES):
        x_next[i] = position[i]
        v_next[i] = velocity[i]
    # 定点迭代
    for _ in ti.static(range(IMPLICIT_ITER)):
        compute_forces(x_next, v_next, f_next)
        for i in range(NUM_PARTICLES):
            if FIXED_PARTICLES[i] == 0:
                v_next[i] = velocity[i] + f_next[i] / MASS * DT
                clamp_vel(v_next, i)
                x_next[i] = position[i] + v_next[i] * DT
    # 写回最终状态
    for i in range(NUM_PARTICLES):
        velocity[i] = v_next[i]
        position[i] = x_next[i]

def main():
    reset_simulation()
    
    # 窗口初始化
    window = ti.ui.Window("Taichi Cloth Simulation", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
    # 相机位置
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    while window.running:
        # GGUI 控制面板
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.42, 0.42)
        window.GUI.text("Integration Method:")
        window.GUI.text("-"*22)
        
        # 按钮选中标记 + 补充特性说明，提升区分度
        prefix0 = "[*] " if current_method[None] == 0 else "[ ] "
        prefix1 = "[*] " if current_method[None] == 1 else "[ ] "
        prefix2 = "[*] " if current_method[None] == 2 else "[ ] "

        if window.GUI.button(prefix0 + "Explicit Euler | 易发散爆炸"):
            current_method[None] = 0

        if window.GUI.button(prefix1 + "Semi-Implicit | 稳定均衡"):
            current_method[None] = 1

        if window.GUI.button(prefix2 + "Implicit Euler | 高稳耗算"):
            current_method[None] = 2


        window.GUI.text("") # 空行分隔
        window.GUI.text("Control Option:")
        window.GUI.text("-"*22)
        # 暂停按钮
        pause_text = "Resume" if paused[None] else "Pause"
        if window.GUI.button(pause_text + " Simulation"):
            paused[None] = 1 - paused[None]
        # 仅此处保留手动重置，需要复位时再点
        if window.GUI.button("Reset Cloth【手动复位】"):
            reset_simulation()
        window.GUI.end()

        # 物理更新
        if not paused[None]:
            # 多子步迭代，流畅模拟
            for _ in range(40):
                if current_method[None] == 0:
                    step_explicit()
                elif current_method[None] == 1:
                    step_semi_implicit()
                else:
                    step_implicit()

        # 渲染
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(position, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(position, indices=spring_line_indices, width=1.5, color=(0.8, 0.8, 0.8))

        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()