import taichi as ti
import taichi.math as tm

# 初始化Taichi CUDA后端（添加debug=True便于调试）
ti.init(arch=ti.cuda, debug=True)

# ===================== 全局常量与类型定义 =====================
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
# 材质ID（用整数常量，避免类型混淆）
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

# 定义相交结果结构体（解决多返回值类型匹配问题）
IntersectResult = ti.types.struct(
    hit=ti.i32,          # 0=False, 1=True（用i32避免布尔类型问题）
    closest_t=ti.f32,
    hit_p=ti.types.vector(3, ti.f32),
    hit_n=ti.types.vector(3, ti.f32),
    mat_id=ti.i32,
    obj_color=ti.types.vector(3, ti.f32),
    is_front_face=ti.i32  # 0=False, 1=True
)

# ===================== 全局字段（严格类型定义） =====================
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(ti.f32, shape=())

# ===================== 斯涅尔定律：折射计算（单一return） =====================
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

# ===================== 光线求交（核心修复：用结构体返回） =====================
@ti.func
def ray_intersect(ray_origin: ti.template(), ray_dir: ti.template()) -> IntersectResult:
    """光线-物体相交检测，返回结构化结果（避免类型不匹配）"""
    # 初始化结果结构体，所有字段显式类型
    res = IntersectResult(
        hit=ti.i32(0),
        closest_t=ti.f32(tm.inf),
        hit_p=ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
        hit_n=ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
        mat_id=ti.i32(MAT_DIFFUSE),
        obj_color=ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
        is_front_face=ti.i32(1)
    )

    # 1. 玻璃球（原红球）
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

    # 2. 镜面球
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

    # 3. 棋盘格地面
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

    return res  # 单一return，符合Taichi 1.7.4要求

# ===================== 硬阴影检测 =====================
@ti.func
def is_shadowed(p: ti.template(), n: ti.template()) -> ti.i32:
    light_dir = light_pos[None] - p
    light_dist = light_dir.norm()
    shadow_ray_dir = light_dir.normalized()
    shadow_ro = p + n * EPS
    res = ray_intersect(shadow_ro, shadow_ray_dir)
    return ti.i32(1) if (res.hit and (res.closest_t < light_dist)) else ti.i32(0)

# ===================== 漫反射着色 =====================
@ti.func
def shade_diffuse(p: ti.template(), n: ti.template(), color: ti.template()) -> ti.template():
    col = AMBIENT * color
    if is_shadowed(p, n) == 0:
        light_dir = (light_pos[None] - p).normalized()
        diffuse = max(n.dot(light_dir), 0.0) * DIFFUSE_POWER
        col += diffuse * color
    return col

# ===================== 渲染主核（修复类型转换） =====================
@ti.kernel
def render():
    camera_pos = ti.Vector([0.0, 0.0, 4.0], dt=ti.f32)
    max_bounce = ti.cast(max_bounces[None], ti.i32)  # 显式转换为整数

    for i, j in pixels:
        final_color = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        
        # MSAA 4倍抗锯齿
        for _ in range(MSAA_SAMPLES):
            # Taichi原生随机数，避免Python random的兼容性问题
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

# ===================== UI交互 =====================
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