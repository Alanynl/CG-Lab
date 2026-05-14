# Taichi Whitted‑Style 光线追踪实验
一个基于Taichi框架实现的迭代式GPU光线追踪演示程序，严格遵循Whitted‑Style全局光照模型，基础功能实现硬阴影、镜面反射、棋盘格地面、交互 UI；选做版本扩展玻璃折射材质（斯涅尔定律 + 全反射）、MSAA多采样抗锯齿，适配Taichi 1.7.4，解决GPU递归限制、自相交阴影噪点、类型崩溃等底层问题。
## 项目介绍
本项目为计算机图形学实验课项目，利用Taichi CUDA并行计算能力实现光线追踪渲染：
1. main.py：基础版光线追踪，实现迭代式光线弹射（替代递归）、硬阴影、镜面反射、黑白棋盘格地面、右上角交互UI；
2. optional.py：选做完整版，在基础版上实现玻璃透明折射材质（斯涅尔定律、全反射）、MSAA 4倍抗锯齿，平滑物体边缘锯齿，提升真实感；

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
    └── Work4/
        ├── __init__.py
        ├── README.md     # 项目说明文档
        ├── main.py       # 基础版：迭代光线追踪+镜面反射+硬阴影+UI交互
        └── optional.py   # 选做完整版：玻璃折射+全反射+MSAA抗锯齿
```
## 文件内容
main.py：基础版光线追踪
核心实现GPU迭代式光线追踪，解决递归GPU不友好问题；实现硬阴影、镜面反射、黑白棋盘格地面、右上角交互面板；

```
import taichi as ti
import math

ti.init(arch=ti.cuda)

WIDTH = 800
HEIGHT = 800
EPS = 1e-4           # 自相交偏移量
REFLECT_RATE = 0.8   # 镜面反射率
AMBIENT = 0.12       # 环境光
DIFFUSE_POWER = 1.2  # 漫反射强度
BG_COLOR = ti.Vector([0.05, 0.15, 0.2])  # 深青蓝色背景
# 材质ID
MAT_DIFFUSE = 0
MAT_MIRROR = 1

# 全局渲染字段 
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(ti.f32, shape=())

# 光线求交
@ti.func
def ray_intersect(ray_origin: ti.template(), ray_dir: ti.template()):
    hit = False
    closest_t = ti.math.inf
    hit_p = ti.Vector([0.0, 0.0, 0.0])
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    mat_id = MAT_DIFFUSE
    obj_color = ti.Vector([0.0, 0.0, 0.0])

    # 红色漫反射球（左球）
    sphere_center = ti.Vector([-1.5, 0.0, 0.0])
    sphere_radius = 1.0
    oc = ray_origin - sphere_center
    a = ray_dir.dot(ray_dir)
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - sphere_radius * sphere_radius
    disc = b * b - 4 * a * c
    if disc > 0:
        t = (-b - ti.sqrt(disc)) / (2.0 * a)
        if EPS < t < closest_t:
            hit = True
            closest_t = t
            hit_p = ray_origin + t * ray_dir
            hit_n = (hit_p - sphere_center).normalized()
            mat_id = MAT_DIFFUSE
            obj_color = ti.Vector([1.0, 0.1, 0.1])

    # 银色镜面球（右球）
    sphere_center2 = ti.Vector([1.5, 0.0, 0.0])
    sphere_radius2 = 1.0
    oc2 = ray_origin - sphere_center2
    a2 = ray_dir.dot(ray_dir)
    b2 = 2.0 * oc2.dot(ray_dir)
    c2 = oc2.dot(oc2) - sphere_radius2 * sphere_radius2
    disc2 = b2 * b2 - 4 * a2 * c2
    if disc2 > 0:
        t = (-b2 - ti.sqrt(disc2)) / (2.0 * a2)
        if EPS < t < closest_t:
            hit = True
            closest_t = t
            hit_p = ray_origin + t * ray_dir
            hit_n = (hit_p - sphere_center2).normalized()
            mat_id = MAT_MIRROR
            obj_color = ti.Vector([0.95, 0.95, 0.95])

    # 地面棋盘格
    plane_y = -1.0
    plane_normal = ti.Vector([0.0, 1.0, 0.0])
    if abs(ray_dir.y) > 1e-6:
        t = (plane_y - ray_origin.y) / ray_dir.y
        if EPS < t < closest_t:
            hit = True
            closest_t = t
            hit_p = ray_origin + t * ray_dir
            hit_n = plane_normal
            mat_id = MAT_DIFFUSE
            check = (int(ti.floor(hit_p.x * 2)) + int(ti.floor(hit_p.z * 2))) % 2
            obj_color = ti.Vector([1.0, 1.0, 1.0]) if check == 0 else ti.Vector([0.1, 0.1, 0.1])

    return hit, closest_t, hit_p, hit_n, mat_id, obj_color

# 硬阴影检测
@ti.func
def is_shadowed(p: ti.template(), n: ti.template()):
    light_dir = light_pos[None] - p
    light_dist = light_dir.norm()
    shadow_ray_dir = light_dir.normalized()
    shadow_ro = p + n * EPS
    hit, t_hit, _, _, _, _ = ray_intersect(shadow_ro, shadow_ray_dir)
    return hit and (t_hit < light_dist)

# 漫反射着色
@ti.func
def shade_diffuse(p: ti.template(), n: ti.template(), color: ti.template()):
    col = AMBIENT * color
    if not is_shadowed(p, n):
        light_dir = (light_pos[None] - p).normalized()
        diffuse = max(n.dot(light_dir), 0.0) * DIFFUSE_POWER
        col += diffuse * color
    return col

# 迭代式光线追踪
@ti.kernel
def render():
    camera_pos = ti.Vector([0.0, 0.0, 4.0])
    max_bounce = int(max_bounces[None])

    for i, j in pixels:
        # 透视投影
        uv = ti.Vector([i / WIDTH, j / HEIGHT]) * 2.0 - 1.0
        uv[0] *= WIDTH / HEIGHT
        ray_dir = ti.Vector([uv[0], uv[1], -1.0]).normalized()
        ray_ro = camera_pos
        ray_rd = ray_dir

        throughput = 1.0
        final_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 弹射计数逻辑
        bounce_times = 0
        while True:
            hit, _, hit_p, hit_n, mat_id, obj_color = ray_intersect(ray_ro, ray_rd)
            if not hit:
                final_color = BG_COLOR * throughput
                break

            # 漫反射材质：着色后直接结束
            if mat_id == MAT_DIFFUSE:
                final_color += throughput * shade_diffuse(hit_p, hit_n, obj_color)
                break

            # 镜面材质 + 最大次数=1：禁止反射，直接显示银色
            if mat_id == MAT_MIRROR:
                if max_bounce == 1:
                    final_color += throughput * obj_color * AMBIENT * 2
                    break
                
                # 镜面材质 + 次数≥2：允许反射，继续弹射
                if bounce_times < max_bounce - 1:
                    reflect_dir = ray_rd - 2 * ray_rd.dot(hit_n) * hit_n
                    reflect_dir = reflect_dir.normalized()
                    ray_ro = hit_p + hit_n * EPS
                    ray_rd = reflect_dir
                    throughput *= REFLECT_RATE
                    bounce_times += 1
                else:
                    # 达到最大弹射次数，停止追踪
                    final_color += throughput * obj_color * AMBIENT
                    break

        pixels[i, j] = final_color

# UI交互
def main():
    # 默认光源位置
    light_pos[None] = [1.571, 4.0, 3.0]
    # 默认3次 → 有反射
    max_bounces[None] = 3.0

    window = ti.ui.Window("Ray Tracing Demo", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        # 右上角控制面板
        with gui.sub_window("Controls", x=0.65, y=0.05, width=0.3, height=0.4):
            light_pos[None][0] = gui.slider_float("Light X", light_pos[None][0], -5.0, 5.0)
            light_pos[None][1] = gui.slider_float("Light Y", light_pos[None][1], -5.0, 5.0)
            light_pos[None][2] = gui.slider_float("Light Z", light_pos[None][2], -5.0, 5.0)
            max_bounces[None] = gui.slider_float("Max Bounces", max_bounces[None], 1.0, 5.0)

        render()
        canvas.set_image(pixels)
        window.show()

if __name__ == "__main__":
    main()
```
## 运行方式
```
uv run -m src.Work4.main
```
## 选做内容
optional.py：选做完整版（折射 + 抗锯齿）
在基础版上实现两项选做加分项：玻璃透明折射材质（斯涅尔定律、全反射判定）、MSAA 4倍抗锯齿；
### 文件内容
```
import taichi as ti
import taichi.math as tm

# 初始化Taichi CUDA后端
ti.init(arch=ti.cuda, debug=True)

# 全局常量与类型定义
WIDTH = 800
HEIGHT = 800
EPS = 1e-4
REFLECT_RATE = 0.8
AMBIENT = 0.12
DIFFUSE_POWER = 1.2
BG_COLOR = ti.Vector([0.05, 0.15, 0.2])  # 1.7.4不支持dtype参数
MSAA_SAMPLES = 4
AIR_IOR = 1.0
GLASS_IOR = 1.5
# 材质ID
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

# 定义相交结果结构体
IntersectResult = ti.types.struct(
    hit=ti.i32,          # 0=False, 1=True（用i32避免布尔类型问题）
    closest_t=ti.f32,
    hit_p=ti.types.vector(3, ti.f32),
    hit_n=ti.types.vector(3, ti.f32),
    mat_id=ti.i32,
    obj_color=ti.types.vector(3, ti.f32),
    is_front_face=ti.i32  # 0=False, 1=True
)

# 全局字段（严格类型定义）
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(ti.f32, shape=())

# 斯涅尔定律：折射计算
@ti.func
def refract(ray_dir: ti.template(), normal: ti.template(), ior_ratio: ti.f32):
    """计算折射光线方向，返回(是否全反射, 方向向量)"""
    is_total_reflect = ti.i32(0)
    result_dir = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
    
    ndotd = normal.dot(ray_dir)
    cos_theta = ti.min(-ndotd, 1.0)
    sin_theta2 = ior_ratio * ior_ratio * (1.0 - cos_theta * cos_theta)
    
    if sin_theta2 > 1.0:
        is_total_reflect = ti.i32(1)
        result_dir = ray_dir - 2 * ndotd * normal
    else:
        refract_perp = ior_ratio * (ray_dir + normal * cos_theta)
        refract_paral = -normal * ti.sqrt(1.0 - sin_theta2)
        result_dir = refract_perp + refract_paral
    
    return is_total_reflect, result_dir

# 光线求交（用结构体返回）
@ti.func
def ray_intersect(ray_origin: ti.template(), ray_dir: ti.template()) -> IntersectResult:
    """光线-物体相交检测，返回结构化结果（避免类型不匹配）"""
    res = IntersectResult(
        hit=ti.i32(0),
        closest_t=ti.f32(tm.inf),
        hit_p=ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
        hit_n=ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
        mat_id=ti.i32(MAT_DIFFUSE),
        obj_color=ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
        is_front_face=ti.i32(1)
    )

    # 玻璃球（原红球）
    sphere_center = ti.Vector([-1.5, 0.0, 0.0], dt=ti.f32)
    sphere_radius = ti.f32(1.0)
    oc = ray_origin - sphere_center
    a = ray_dir.dot(ray_dir)
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - sphere_radius * sphere_radius
    disc = b * b - 4 * a * c
    if disc > 0:
        t = (-b - ti.sqrt(disc)) / (2.0 * a)
        if EPS < t < res.closest_t:
            res.hit = ti.i32(1)
            res.closest_t = t
            res.hit_p = ray_origin + t * ray_dir
            res.hit_n = (res.hit_p - sphere_center).normalized()
            # 显式转换为i32，避免精度警告
            res.is_front_face = ti.i32(1) if (ray_dir.dot(res.hit_n) < 0) else ti.i32(0)
            if res.is_front_face == 0:
                res.hit_n = -res.hit_n
            res.mat_id = ti.i32(MAT_GLASS)
            res.obj_color = ti.Vector([0.95, 0.98, 1.0], dt=ti.f32)

    # 镜面球
    sphere_center2 = ti.Vector([1.5, 0.0, 0.0], dt=ti.f32)
    sphere_radius2 = ti.f32(1.0)
    oc2 = ray_origin - sphere_center2
    a2 = ray_dir.dot(ray_dir)
    b2 = 2.0 * oc2.dot(ray_dir)
    c2 = oc2.dot(oc2) - sphere_radius2 * sphere_radius2
    disc2 = b2 * b2 - 4 * a2 * c2
    if disc2 > 0:
        t = (-b2 - ti.sqrt(disc2)) / (2.0 * a2)
        if EPS < t < res.closest_t:
            res.hit = ti.i32(1)
            res.closest_t = t
            res.hit_p = ray_origin + t * ray_dir
            res.hit_n = (res.hit_p - sphere_center2).normalized()
            res.mat_id = ti.i32(MAT_MIRROR)
            res.obj_color = ti.Vector([0.95, 0.95, 0.95], dt=ti.f32)

    # 棋盘格地面
    plane_y = ti.f32(-1.0)
    plane_normal = ti.Vector([0.0, 1.0, 0.0], dt=ti.f32)
    if abs(ray_dir.y) > 1e-6:
        t = (plane_y - ray_origin.y) / ray_dir.y
        if EPS < t < res.closest_t:
            res.hit = ti.i32(1)
            res.closest_t = t
            res.hit_p = ray_origin + t * ray_dir
            res.hit_n = plane_normal
            res.mat_id = ti.i32(MAT_DIFFUSE)
            # 显式转换为整数，避免类型警告
            check = (ti.cast(ti.floor(res.hit_p.x * 2), ti.i32) + 
                    ti.cast(ti.floor(res.hit_p.z * 2), ti.i32)) % 2
            if check == 0:
                res.obj_color = ti.Vector([1.0, 1.0, 1.0], dt=ti.f32)
            else:
                res.obj_color = ti.Vector([0.1, 0.1, 0.1], dt=ti.f32)

    return res

# 硬阴影检测
@ti.func
def is_shadowed(p: ti.template(), n: ti.template()) -> ti.i32:
    light_dir = light_pos[None] - p
    light_dist = light_dir.norm()
    shadow_ray_dir = light_dir.normalized()
    shadow_ro = p + n * EPS
    res = ray_intersect(shadow_ro, shadow_ray_dir)
    return ti.i32(1) if (res.hit and (res.closest_t < light_dist)) else ti.i32(0)

# 漫反射着色
@ti.func
def shade_diffuse(p: ti.template(), n: ti.template(), color: ti.template()) -> ti.template():
    col = AMBIENT * color
    if is_shadowed(p, n) == 0:
        light_dir = (light_pos[None] - p).normalized()
        diffuse = max(n.dot(light_dir), 0.0) * DIFFUSE_POWER
        col += diffuse * color
    return col

# 渲染主核
@ti.kernel
def render():
    camera_pos = ti.Vector([0.0, 0.0, 4.0], dt=ti.f32)
    max_bounce = ti.cast(max_bounces[None], ti.i32)  # 显式转换为整数

    for i, j in pixels:
        final_color = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        
        # MSAA 4倍抗锯齿
        for _ in range(MSAA_SAMPLES):
            du = (ti.random(ti.f32)-0.5) / ti.cast(WIDTH, ti.f32)
            dv = (ti.random(ti.f32)-0.5) / ti.cast(HEIGHT, ti.f32)
            uv_x = (ti.cast(i, ti.f32) + du) / ti.cast(WIDTH, ti.f32)
            uv_y = (ti.cast(j, ti.f32) + dv) / ti.cast(HEIGHT, ti.f32)
            uv = ti.Vector([uv_x, uv_y], dt=ti.f32)*2.0-1.0
            uv[0] *= ti.cast(WIDTH, ti.f32) / ti.cast(HEIGHT, ti.f32)
            ray_dir = ti.Vector([uv[0], uv[1], -1.0], dt=ti.f32).normalized()
            
            ray_ro = camera_pos
            ray_rd = ray_dir
            throughput = ti.f32(1.0)
            sample_color = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            bounce_times = ti.i32(0)

            while True:
                # 调用相交检测，接收结构化结果
                res = ray_intersect(ray_ro, ray_rd)
                
                if res.hit == 0:
                    sample_color = BG_COLOR * throughput
                    break

                # 漫反射材质
                if res.mat_id == MAT_DIFFUSE:
                    sample_color += throughput * shade_diffuse(res.hit_p, res.hit_n, res.obj_color)
                    break

                # 镜面材质
                elif res.mat_id == MAT_MIRROR:
                    if max_bounce == 1:
                        sample_color += throughput * res.obj_color * AMBIENT * 2
                        break
                    if bounce_times < max_bounce - 1:
                        reflect_dir = ray_rd - 2 * ray_rd.dot(res.hit_n) * res.hit_n
                        ray_ro = res.hit_p + res.hit_n * EPS
                        ray_rd = reflect_dir.normalized()
                        throughput *= REFLECT_RATE
                        bounce_times += 1
                    else:
                        sample_color += throughput * res.obj_color * AMBIENT
                        break

                # 玻璃材质（折射+全反射）
                elif res.mat_id == MAT_GLASS:
                    if bounce_times >= max_bounce - 1:
                        sample_color += throughput * res.obj_color * AMBIENT
                        break
                    # 计算折射率比值
                    ior = ti.f32(GLASS_IOR / AIR_IOR) if res.is_front_face == 1 else ti.f32(AIR_IOR / GLASS_IOR)
                    # 计算折射/全反射
                    is_total_reflect, new_dir = refract(ray_rd, res.hit_n, ior)
                    # 更新光线
                    ray_ro = res.hit_p + res.hit_n * EPS
                    ray_rd = new_dir.normalized()
                    throughput *= REFLECT_RATE
                    bounce_times += 1

            final_color += sample_color

        # 所有采样结果取平均
        pixels[i, j] = final_color / ti.cast(MSAA_SAMPLES, ti.f32)

# UI交互
def main():
    # 默认光源位置和弹射次数
    light_pos[None] = ti.Vector([1.571, 4.0, 3.0], dt=ti.f32)
    max_bounces[None] = 3.0  # 3次弹射 → 有反射/折射

    window = ti.ui.Window("Ray Tracing (Glass + AA)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        with gui.sub_window("Controls", x=0.65, y=0.05, width=0.3, height=0.4):
            light_pos[None][0] = gui.slider_float("Light X", light_pos[None][0], -5.0, 5.0)
            light_pos[None][1] = gui.slider_float("Light Y", light_pos[None][1], -5.0, 5.0)
            light_pos[None][2] = gui.slider_float("Light Z", light_pos[None][2], -5.0, 5.0)
            max_bounces[None] = gui.slider_float("Max Bounces", max_bounces[None], 1.0, 5.0)

        render()
        canvas.set_image(pixels)
        window.show()

if __name__ == "__main__":
    main()
```
### 运行方式
```
uv run -m src.Work4.optional
```
### 优化点
1. 选做 1：玻璃折射材质
- 左侧红球改造为透明玻璃材质，实现斯涅尔定律折射计算；
- 自动判定全反射：光线在玻璃内部入射角过大时，触发全反射效果；
- 光线穿透玻璃，可观察到背后棋盘格、镜面球的扭曲折射影像，物理真实；
- 适配迭代式光线弹射，支持多次折射/反射叠加。
2. 选做 2：MSAA 抗锯齿
- 每个像素内随机发射4条亚像素主光线，颜色取平均值；
- 彻底消除球体、地面边缘的锯齿像素，画面边缘平滑自然；
- 平衡性能与画质，4倍采样兼顾GPU渲染速度与视觉效果。

## 交互说明
- 滑动条 Light X/Y/Z：调整点光源三维坐标，实时观察阴影位置变化
- 滑动条 Max Bounces：最大光线弹射次数
- 数值 = 1：关闭所有反射/折射，镜面球、玻璃球仅显示纯色
- 数值≥2：开启镜面反射、玻璃折射，弹射次数越多反射层数越丰富
- 默认值 = 3：正常开启反射/折射效果
- ESC 键：关闭渲染窗口
- 实时GPU渲染，拖动滑块画面即时更新

## 自定义参数
```
WIDTH, HEIGHT = 800, 800    # 渲染窗口分辨率
EPS = 1e-4                   # 自相交偏移量（解决阴影噪点）
REFLECT_RATE = 0.8           # 镜面反射衰减系数
AMBIENT = 0.12               # 环境光强度
MSAA_SAMPLES = 4             # 抗锯齿采样数
AIR_IOR = 1.0, GLASS_IOR=1.5 # 空气、玻璃折射率
# 材质颜色
COLOR_RED_SPHERE = (1.0,0.1,0.1)
COLOR_MIRROR = (0.95,0.95,0.95)
COLOR_GLASS = (0.95,0.98,1.0)
```
## 常见问题
### 1. CUDA启动失败/报错
- 显卡不支持 CUDA：将 ti.init(arch=ti.cuda) 改为 ti.init(arch=ti.cpu)，使用 CPU 渲染；
- Taichi 版本不匹配：严格安装 taichi==1.7.4，新版语法不兼容；
### 2. 满屏黑色噪点（Shadow Acne）
- 检查 EPS=1e-4 是否正确，次级射线（阴影/反射）必须沿法线偏移极小值，避免光线与自身相交；
### 3. 镜面无反射效果
- 检查Max Bounces数值：必须大于1，1次弹射仅显示物体本身；
- 确认光线迭代逻辑：镜面材质必须更新光线起点、方向，继续循环弹射；
### 4. 渲染卡顿
- 降低MSAA采样数（改为2），或降低窗口分辨率，减少GPU计算量。
