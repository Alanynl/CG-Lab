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