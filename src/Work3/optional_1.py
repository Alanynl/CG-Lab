import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

# 工具函数
@ti.func
def normalize(v):
    return v / v.norm(1e-5)

# 基础参数
vec3 = ti.types.vector(3, ti.f32)
width, height = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

# 场景配置
cam_pos = vec3(0.0, 0.0, 5.0)
light_pos = vec3(2.0, 3.0, 4.0)
light_color = vec3(1.0, 1.0, 1.0)

sphere_center = vec3(-1.2, -0.2, 0.0)
sphere_radius = 1.2
sphere_color = vec3(0.8, 0.1, 0.1)

cone_vertex = vec3(1.2, 1.2, 0.0)
cone_base_y = -1.4
cone_base_radius = 1.2
cone_color = vec3(0.6, 0.2, 0.8)

# 材质参数
Ka = ti.field(dtype=ti.f32, shape=())
Kd = ti.field(dtype=ti.f32, shape=())
Ks = ti.field(dtype=ti.f32, shape=())
shininess = ti.field(dtype=ti.f32, shape=())

Ka[None] = 0.2
Kd[None] = 0.7
Ks[None] = 0.5
shininess[None] = 32.0

# 球体求交
@ti.func
def ray_sphere_intersect(origin: vec3, dir: vec3, center: vec3, radius: ti.f32):
    t = -1.0
    normal = vec3(0.0, 0.0, 0.0)
    oc = origin - center
    b = 2.0 * oc.dot(dir)
    c = oc.dot(oc) - radius**2
    delta = b*b - 4.0*c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 1e-4:
            t = t1
            p = origin + dir * t
            normal = normalize(p - center)
    return t, normal

# 圆锥求交（修复法向量，光照正常）
@ti.func
def ray_cone_intersect(origin: vec3, dir: vec3):
    t = -1.0
    normal = vec3(0.0, 0.0, 0.0)
    H = cone_vertex.y - cone_base_y
    k = (cone_base_radius / H) ** 2
    
    ro_local = origin - cone_vertex
    A = dir.x**2 + dir.z**2 - k * dir.y**2
    B = 2.0 * (ro_local.x * dir.x + ro_local.z * dir.z - k * ro_local.y * dir.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    
    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0*A*C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0*A)
            t2 = (-B + ti.sqrt(delta)) / (2.0*A)
            t_first = t1 if t1 < t2 else t2
            t_second = t2 if t1 < t2 else t1
            
            y1 = ro_local.y + t_first * dir.y
            if t_first > 0 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * dir.y
                if t_second > 0 and -H <= y2 <= 0:
                    t = t_second
                    
            if t > 0:
                p_local = ro_local + dir * t
                normal = normalize(vec3(p_local.x, -k * p_local.y, p_local.z))
    return t, normal

# 渲染内核（Blinn-Phong 核心）
@ti.kernel
def render():
    for i, j in pixels:
        u = (i - width / 2.0) / height * 2.0
        v = (j - height / 2.0) / height * 2.0
        
        dir = normalize(vec3(u, v, -1.0))
        origin = cam_pos

        t_sph, n_sph = ray_sphere_intersect(origin, dir, sphere_center, sphere_radius)
        t_cone, n_cone = ray_cone_intersect(origin, dir)
        
        min_t = 1e10
        hit_normal = vec3(0.0, 0.0, 0.0)
        hit_color = vec3(0.0, 0.0, 0.0)
        
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = sphere_color
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = cone_color

        color = vec3(0.05, 0.15, 0.15)
        
        if min_t < 1e9:
            p = origin + dir * min_t
            N = hit_normal
            L = normalize(light_pos - p)
            V = normalize(origin - p)

            ambient = Ka[None] * light_color * hit_color
            diffuse = Kd[None] * ti.max(0.0, N.dot(L)) * light_color * hit_color
            
            # Blinn-Phong 高光（半程向量）
            H = normalize(L + V)
            specular = Ks[None] * ti.max(0.0, N.dot(H)) ** shininess[None] * light_color
            
            color = ambient + diffuse + specular
        
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

# 主窗口
def main():
    window = ti.ui.Window("选做1：Blinn-Phong", (width, height))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("参数", 0.55, 0.1, 0.4, 0.4) as w:
            Ka[None] = w.slider_float("Ka", Ka[None], 0.0, 1.0)
            Kd[None] = w.slider_float("Kd", Kd[None], 0.0, 1.0)
            Ks[None] = w.slider_float("Ks", Ks[None], 0.0, 1.0)
            shininess[None] = w.slider_float("高光", shininess[None], 1.0, 128.0)

        window.show()

if __name__ == "__main__":
    main()