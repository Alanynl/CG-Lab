import taichi as ti

# 初始化Taichi（GPU加速）
ti.init(arch=ti.gpu, default_fp=ti.f32, debug=False)

# 全局参数
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 20  # 布料网格尺寸
NUM_PARTICLES = GRID_SIZE * GRID_SIZE
DT = 5e-4
MASS = 1.0
KS = 10000.0  # 弹簧劲度系数
KD = 1.0      # 阻尼系数
GRAVITY = ti.Vector([0.0, -9.8, 0.0])
MAX_VELOCITY = 50.0
IMPLICIT_ITER = 3

# 球体参数（1D场，适配particles渲染）
sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=(1,))  # shape=(1,) 1D场
sphere_radius = ti.field(dtype=ti.f32, shape=())  # 半径保持0D场
collision_on = ti.field(dtype=int, shape=())  # 碰撞开关

# 物理场
position = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
velocity = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
force = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
FIXED_PARTICLES = ti.field(dtype=int, shape=NUM_PARTICLES)

# 隐式欧拉缓存
x_next = ti.Vector.field(3, ti.f32, NUM_PARTICLES)
v_next = ti.Vector.field(3, ti.f32, NUM_PARTICLES)
f_next = ti.Vector.field(3, ti.f32, NUM_PARTICLES)

# 弹簧系统
max_springs = GRID_SIZE * GRID_SIZE * 8
num_springs = ti.field(dtype=int, shape=())
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_rest_length = ti.field(dtype=ti.f32, shape=max_springs)
spring_line_indices = ti.field(dtype=int, shape=max_springs * 2)

# 控制参数
current_method = ti.field(dtype=int, shape=())
paused = ti.field(dtype=int, shape=())

# 初始化
@ti.kernel
def init_positions():
    for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
        idx = i * GRID_SIZE + j
        position[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        velocity[idx] = ti.Vector([0.0, 0.0, 0.0])
        force[idx] = ti.Vector([0.0, 0.0, 0.0])
        FIXED_PARTICLES[idx] = 1 if (j == 0 and (i == 0 or i == GRID_SIZE - 1)) else 0

@ti.kernel
def init_springs():
    """完整弹簧：结构+剪切+弯曲"""
    num_springs[None] = 0
    for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
        idx = i * GRID_SIZE + j
        
        # 1. 结构弹簧（原有）- 上下左右
        if i < GRID_SIZE - 1:
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, (i+1)*GRID_SIZE+j])
            spring_rest_length[c] = (position[idx] - position[(i+1)*GRID_SIZE+j]).norm()
        if j < GRID_SIZE - 1:
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, i*GRID_SIZE+(j+1)])
            spring_rest_length[c] = (position[idx] - position[i*GRID_SIZE+(j+1)]).norm()
        
        # 2. 剪切弹簧（新增）- 对角线
        if i < GRID_SIZE - 1 and j < GRID_SIZE - 1:
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, (i+1)*GRID_SIZE+(j+1)])
            spring_rest_length[c] = (position[idx] - position[(i+1)*GRID_SIZE+(j+1)]).norm()
        if i > 0 and j < GRID_SIZE - 1:
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, (i-1)*GRID_SIZE+(j+1)])
            spring_rest_length[c] = (position[idx] - position[(i-1)*GRID_SIZE+(j+1)]).norm()
        
        # 3. 弯曲弹簧（新增）- 隔一个质点
        if i < GRID_SIZE - 2:
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, (i+2)*GRID_SIZE+j])
            spring_rest_length[c] = (position[idx] - position[(i+2)*GRID_SIZE+j]).norm()
        if j < GRID_SIZE - 2:
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, i*GRID_SIZE+(j+2)])
            spring_rest_length[c] = (position[idx] - position[i*GRID_SIZE+(j+2)]).norm()

@ti.kernel
def init_spring_lines():
    for i in range(num_springs[None]):
        spring_line_indices[i*2] = spring_pairs[i][0]
        spring_line_indices[i*2+1] = spring_pairs[i][1]

def reset_simulation():
    """重置整个模拟状态"""
    sphere_center[0] = ti.Vector([0.0, 0.2, 0.0])  # 1D场用索引0访问
    sphere_radius[None] = 0.25
    collision_on[None] = 1  # 默认开启碰撞
    
    init_positions()
    init_springs()
    init_spring_lines()
    current_method[None] = 1
    paused[None] = 0
    print("Simulation reset!")

# 物理计算函数
@ti.func
def compute_forces(pos: ti.template(), vel: ti.template(), f: ti.template()):
    """计算所有力（重力+阻尼+弹簧力）"""
    for i in range(NUM_PARTICLES):
        f[i] = GRAVITY * MASS - KD * vel[i]
    for i in range(num_springs[None]):
        a, b = spring_pairs[i][0], spring_pairs[i][1]
        dx = pos[a] - pos[b]
        dist = dx.norm()
        if dist > 1e-6:
            sf = -KS * (dist - spring_rest_length[i]) * (dx / dist)
            ti.atomic_add(f[a], sf)
            ti.atomic_add(f[b], -sf)

@ti.func
def clamp_vel(vel: ti.template(), idx: int):
    """速度钳制，防止数值爆炸"""
    speed = vel[idx].norm()
    if speed > MAX_VELOCITY:
        vel[idx] = vel[idx] / speed * MAX_VELOCITY

@ti.func
def handle_collision_single(pos: ti.template(), vel: ti.template(), idx: int, 
                          center: ti.template(), radius: ti.f32):
    if FIXED_PARTICLES[idx] == 0:
        diff = pos[idx] - center
        dist = diff.norm()
        # 质点进入球体内部，修正位置+反弹
        if dist < radius:
            n = diff / dist
            pos[idx] = center + n * radius
            # 法向速度清零（简单弹性碰撞）
            vn = vel[idx].dot(n)
            if vn < 0:
                vel[idx] -= vn * n

@ti.func
def handle_collision_batch(pos: ti.template(), vel: ti.template()):
    """批量碰撞处理（ti.func内联，遍历所有质点）"""
    if collision_on[None] == 1:
        center = sphere_center[0]  # 获取球体中心
        radius = sphere_radius[None]
        for i in range(NUM_PARTICLES):
            handle_collision_single(pos, vel, i, center, radius)

# 三种积分器
@ti.kernel
def step_explicit():
    """显式欧拉（Explicit Euler）- 极易发散"""
    compute_forces(position, velocity, force)
    for i in range(NUM_PARTICLES):
        if FIXED_PARTICLES[i] == 0:
            position[i] += velocity[i] * DT
            velocity[i] += force[i] / MASS * DT
            clamp_vel(velocity, i)
    handle_collision_batch(position, velocity)  # 内联碰撞处理（ti.func）

@ti.kernel
def step_semi_implicit():
    """半隐式欧拉（Semi-Implicit Euler）- 相对稳定"""
    compute_forces(position, velocity, force)
    for i in range(NUM_PARTICLES):
        if FIXED_PARTICLES[i] == 0:
            velocity[i] += force[i] / MASS * DT
            clamp_vel(velocity, i)
            position[i] += velocity[i] * DT
    handle_collision_batch(position, velocity)  # 内联碰撞处理（ti.func）

@ti.kernel
def step_implicit():
    """隐式欧拉（Implicit Euler）- 使用定点迭代法近似求解"""
    # 复制当前状态到预测场
    for i in range(NUM_PARTICLES):
        x_next[i] = position[i]
        v_next[i] = velocity[i]
    # 定点迭代求解未来状态（ti.static 在编译期展开，无循环开销）
    for _ in ti.static(range(IMPLICIT_ITER)):
        compute_forces(x_next, v_next, f_next)
        for i in range(NUM_PARTICLES):
            if FIXED_PARTICLES[i] == 0:
                v_next[i] = velocity[i] + f_next[i] / MASS * DT
                clamp_vel(v_next, i)
                x_next[i] = position[i] + v_next[i] * DT
    # 将收敛后的状态写回
    for i in range(NUM_PARTICLES):
        velocity[i] = v_next[i]
        position[i] = x_next[i]
    handle_collision_batch(position, velocity)  # 内联碰撞处理（ti.func）

def main():
    reset_simulation()
    
    window = ti.ui.Window("Taichi Cloth (完整弹簧+球体碰撞)", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    while window.running:
        # GUI控制面板
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.42, 0.45)
        window.GUI.text("=== Integration Method ===")
        prefix0 = "[*] " if current_method[None] == 0 else "[ ] "
        prefix1 = "[*] " if current_method[None] == 1 else "[ ] "
        prefix2 = "[*] " if current_method[None] == 2 else "[ ] "

        if window.GUI.button(prefix0 + "Explicit | 易爆炸"):
            current_method[None] = 0
        if window.GUI.button(prefix1 + "Semi-Implicit | 稳定"):
            current_method[None] = 1
        if window.GUI.button(prefix2 + "Implicit | 超稳定"):
            current_method[None] = 2

        window.GUI.text("=== Control ===")
        pause_text = "Resume" if paused[None] else "Pause"
        if window.GUI.button(pause_text + " Simulation"):
            paused[None] = 1 - paused[None]
        
        # 碰撞开关
        col_text = "Collision: ON" if collision_on[None] else "Collision: OFF"
        if window.GUI.button(col_text):
            collision_on[None] = 1 - collision_on[None]
        
        if window.GUI.button("Reset Cloth"):
            reset_simulation()
        window.GUI.end()

        # 物理更新
        if not paused[None]:
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
        scene.point_light((0.5, 1.5, 1.5), (1, 1, 1))
        
        # 绘制布料
        scene.particles(position, 0.015, (0.2, 0.6, 1.0))
        scene.lines(position, indices=spring_line_indices, width=1.5, color=(0.8, 0.8, 0.8))
        
        # 正确渲染球体
        scene.particles(sphere_center, radius=sphere_radius[None], color=(1, 0.3, 0.3))
        
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()