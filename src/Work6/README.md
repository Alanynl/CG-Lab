# Taichi 质点-弹簧布料物理模拟实验
一个基于Taichi框架实现GPU并行布料仿真程序，采用胡克定律+阻尼构建质点弹簧系统，基础功能实现结构弹簧、三种欧拉数值积分、实时GUI控制面板；选做版本扩展剪切+弯曲双类型弹簧、球体碰撞检测，适配Taichi 1.7.4，解决显式欧拉数值爆炸、GPU并行原子写入、Kernel嵌套报错等问题。
## 项目介绍
本项目为计算机图形学实验课项目，利用Taichi CUDA并行计算实现布料动态下落仿真：
1. main.py：基础版布料仿真，仅结构弹簧，实现显式/半隐式/隐式三种积分、速度钳制防爆、右上角交互UI；
2. optional.py：选做完整版，新增剪切弹簧+弯曲弹簧，加入球体碰撞物理检测，可一键开关碰撞效果；
## 技术栈
- Python 3.12+
- Taichi 1.7.4
- CUDA/CPU 后端
- uv
- Git
## 环境准备
### 安装配置
```
 powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
 uv init --python 3.12
 uv sync
 uv add taichi
```
## 项目结构
```
CG-Lab/
│
├── pyproject.toml        # 项目依赖管理文件
└── src/
    └── Work6/
        ├── __init__.py
        ├── README.md     # 项目说明文档
        ├── main.py       # 基础版：结构弹簧+三种欧拉积分+UI交互
        └── optional.py   # 选做完整版：剪切+弯曲弹簧+球体碰撞
```
## 文件内容
main.py：基础版布料质点弹簧仿真
基础功能：水平+垂直结构弹簧、三种数值积分（显式/半隐式/定点迭代隐式欧拉）、速度限幅防止数值爆炸、GUI切换算法，采用拆分Kernel初始化保证GPU同步，ti.func封装受力与限速函数减少GPU开销。
```
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
```
## 运行方式
```
uv run -m src.Work6.main
```
## 选做内容
optional.py：选做完整版（剪切+弯曲全弹簧 + 球体碰撞检测）
在基础版上拓展两类额外弹簧、实现布料与球体碰撞，GUI增加碰撞开关，碰撞采用质点距离判定+位置修正防穿透。
```
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
```
### 运行方式
```
uv run -m src.Work6.optional
```
### 优化点
1. 选做1：三种弹簧拓展
- 原有结构弹簧+对角线剪切弹簧+隔点弯曲弹簧；
- 剪切弹簧防止网格对角撕裂，弯曲弹簧抑制布料软塌，布料仿真更贴近真实织物；
- 三种弹簧统一遍历受力计算，使用原子加法规避GPU多线程写入冲突。
2. 选做2：球体碰撞系统
- 场景中心生成红色碰撞球体，一键开关碰撞；
- 开启碰撞：布料下落贴合球面、不会穿入球体；关闭碰撞：布料直接穿透球体自由下落；
- 碰撞修正质点位置+法向速度校正，无质点穿透BUG。

## 交互说明
- 三个积分按钮：实时切换求解器，切换不重置布料状态
  - Explicit：显式欧拉，大刚度极易抖动爆炸
  - Semi-Implicit：半隐式，默认稳定
  - Implicit：隐式欧拉，稳定性最强，摆动衰减更快
- Pause/Resume：暂停/继续仿真
- Collision开关（仅optional）：开启/关闭球体碰撞
- Reset Cloth：一键重置布料回到初始悬挂状态
- 相机：右键按住拖动旋转视角、滚轮缩放画面

## 自定义参数
```
WIDTH, HEIGHT = 800, 800    # 窗口分辨率
GRID_SIZE = 20               # 布料网格行列数
DT = 5e-4                    # 单步时间
MASS = 1.0                   # 单个质点质量
KS = 10000.0                 # 弹簧劲度
KD = 1.0                     # 阻尼系数
GRAVITY = [0, -9.8, 0]       # 重力矢量
MAX_VELOCITY = 50.0          # 速度上限（防爆）
IMPLICIT_ITER = 3            # 隐式迭代次数
SPHERE_R = 0.25              # 碰撞球半径
```
## 常见问题
### 1. CUDA启动失败
改为`ti.init(arch=ti.cpu)`使用CPU运行，taichi版本锁定1.7.4。
### 2. 显式欧拉布料乱飞爆炸
减小DT时间步、降低KS刚度、提高阻尼KD，或切换半隐/隐式算法。
### 3 布料穿透红球
检查碰撞函数位置修正逻辑，球体中心与半径参数。
### 4 运行卡顿
减小GRID网格尺寸、减少每帧子步步数。