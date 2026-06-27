- 姓名：陈蓝薪
- 学号：202411998405
- 专业：人工智能
# Taichi 光线追踪与光照渲染实验
基于Taichi框架实现的3D光线追踪渲染实验，依托GPU并行加速完成球体/圆锥光线求交、Phong/Blinn-Phong光照模型、硬阴影渲染，支持交互式材质参数调节，直观展示计算机图形学光照渲染核心原理。
## 项目介绍
本项目为计算机图形学光照渲染实验，使用Taichi实现高性能3D场景光线追踪，包含三个递进版本，完整覆盖基础光照到阴影渲染的核心知识点：
1. main.py：基础版，实现经典Phong光照模型，完成光线 - 球体/圆锥求交与材质渲染，支持交互式参数调节。
2. optional_1.py：优化版，改用Blinn-Phong光照模型，通过半程向量优化高光计算，渲染效率更高、高光效果更柔和。
3. optional_2.py：进阶版，在Phong光照基础上新增硬阴影检测，实现物体间光照遮挡效果，还原真实光影关系。
4. 全版本均支持GPU加速渲染，内置交互式GUI面板，可实时调整环境光、漫反射、高光参数，即时查看渲染效果变化。

## 技术栈
- Python 3.8+
- Taichi 1.6.0+
- uv
- Git
- 核心算法：光线追踪、Phong 光照、Blinn-Phong 光照、硬阴影检测

## 环境准备
### 安装依赖
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
    └── Work3/
        ├── __init__.py
        ├── README.md     # 项目说明文档
        ├── main.py       # 基础版：Phong 光照渲染
        ├── optional_1.py # 选做1：Blinn-Phong 光照渲染
        └── optional_2.py # 选做2：Phong 光照+硬阴影渲染
```
## 文件内容
### main.py（基础Phong光照）
- 核心实现：经典Phong光照模型（环境光 + 漫反射 + 镜面反射）
- 场景元素：红色球体、紫色圆锥，固定相机与点光源
```
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

# 自定义工具函数
@ti.func
def normalize(v):
    return v / v.norm(1e-5)

# 向量定义&画布参数
vec3 = ti.types.vector(3, ti.f32)
width, height = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

# 场景参数
cam_pos = vec3(0.0, 0.0, 5.0)
light_pos = vec3(2.0, 3.0, 4.0)
light_color = vec3(1.0, 1.0, 1.0)

# 红色球体
sphere_center = vec3(-1.2, -0.2, 0.0)
sphere_radius = 1.2
sphere_color = vec3(0.8, 0.1, 0.1)

# 紫色圆锥
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

# 光线求交函数
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

# 主渲染内核
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
            
            R = normalize(-L - 2.0 * (-L).dot(N) * N)
            specular = Ks[None] * ti.max(0.0, R.dot(V)) ** shininess[None] * light_color
            
            color = ambient + diffuse + specular
        
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

# UI与主循环
def main():
    window = ti.ui.Window("Phong Shading Demo", (width, height))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Material Parameters", 0.55, 0.1, 0.4, 0.4) as w:
            w.text("Phong Shading Parameters")
            Ka[None] = w.slider_float("Ka", Ka[None], 0.0, 1.0)
            Kd[None] = w.slider_float("Kd", Kd[None], 0.0, 1.0)
            Ks[None] = w.slider_float("Ks", Ks[None], 0.0, 1.0)
            shininess[None] = w.slider_float("Shininess", shininess[None], 1.0, 128.0)

        window.show()

if __name__ == "__main__":
    main()
```
#### 关键功能：
- 光线 - 球体/圆锥精确求交
- 基于反射向量计算高光，还原标准Phong渲染效果
- 交互式 GUI 调节 Ka/Kd/Ks/ 高光指数参数
- 渲染效果：基础3D物体光照着色，无阴影效果

## 运行方式
```
uv run -m src.Work3.main
```
## 演示视频
<img width="480" height="507" alt="Work3_main" src="https://github.com/user-attachments/assets/84facbb5-8c61-426a-910f-64ccb7873ccc" />

## 选做内容
### 选做1： optional_1.py（Blinn-Phong:优化光照）
核心优化：将Phong高光替换为Blinn-Phong半程向量高光
#### 改进点：
- 省去反射向量计算，渲染效率更高
- 高光效果更柔和，更贴合真实物理光照
- 保留所有交互功能，UI 文案优化为中文
- 适用场景：追求更高渲染效率与更自然高光的光照演示
#### 文件内容
```
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

# 圆锥求交
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
```
#### 演示视频
<img width="480" height="507" alt="Work3_op1" src="https://github.com/user-attachments/assets/64bd86b5-3447-4ed0-a899-e0b37d40d087" />

### 选做2：optional_2.py（Phong 光照 + 硬阴影）
核心新增：硬阴影检测函数:shadow_check
#### 关键实现：
- 从物体交点向光源发射阴影光线，判断是否被遮挡
- 阴影区域仅保留环境光，非阴影区域正常计算全光照
- 基于传统 Phong 光照，实现清晰的硬阴影边缘
- 渲染效果：球体与圆锥相互遮挡产生硬阴影，场景立体感与真实感大幅提升

#### 文件内容
```
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

# 阴影检测函数
@ti.func
def shadow_check(p: vec3, N: vec3):
    shadow_ro = p + N * 1e-4
    shadow_dir = normalize(light_pos - p)
    t_light = (light_pos - p).norm()
    
    t1, _ = ray_sphere_intersect(shadow_ro, shadow_dir, sphere_center, sphere_radius)
    t2, _ = ray_cone_intersect(shadow_ro, shadow_dir)
    
    min_t = 1e10
    if t1 > 0: min_t = t1
    if t2 > 0 and t2 < min_t: min_t = t2
    return 0 < min_t < t_light

# 渲染内核（硬阴影核心）
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

            # 硬阴影判断
            in_shadow = shadow_check(p, N)
            
            ambient = Ka[None] * light_color * hit_color
            if in_shadow:
                color = ambient
            else:
                diffuse = Kd[None] * ti.max(0.0, N.dot(L)) * light_color * hit_color
                R = normalize(-L - 2.0 * (-L).dot(N) * N)
                specular = Ks[None] * ti.max(0.0, R.dot(V)) ** shininess[None] * light_color
                color = ambient + diffuse + specular
        
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

# 主窗口
def main():
    window = ti.ui.Window("选做2：硬阴影", (width, height))
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
```
#### 演示视频
<img width="480" height="507" alt="Work3_op2" src="https://github.com/user-attachments/assets/2b280155-f975-4eb1-bd56-15f42075f404" />

## 交互说明
- 程序启动后自动渲染3D场景（红色球体 + 紫色圆锥）
- 右侧GUI面板参数调节：
-- Ka：环境光系数，控制物体基础亮度
-- Kd：漫反射系数，控制物体表面漫反射强度
-- Ks：高光系数，控制物体高光亮度
-- Shininess / 高光：高光指数，控制高光区域大小（值越大高光越集中）
- 关闭窗口退出程序

## 核心技术原理
### 光线求交
- 球体：解二次方程计算光线与球体交点，获取法向量
- 圆锥：局部坐标系转换 + 二次方程求解，限制交点范围并修正法向量
### 光照模型
- Phong：环境光 + 漫反射 + 反射向量高光
- Blinn-Phong：环境光 + 漫反射 + 半程向量高光（效率优化）
### 硬阴影
- 阴影光线检测：判断交点到光源的路径是否被几何体遮挡，实现二元阴影（有/无阴影）

## 常见问题
### 运行报错/窗口崩溃
解决方案：将 ti.init(arch=ti.gpu) 改为 ti.init(arch=ti.cpu) 使用 CPU 渲染
### 无渲染画面/物体不显示
解决方案：更新 Taichi 至最新版，确保 Python 版本 ≥3.8
### 参数调节无效果
解决方案：拖动滑块时保持窗口选中，确保参数在 0~1 合理范围内调整
