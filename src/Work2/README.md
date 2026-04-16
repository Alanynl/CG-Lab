# Taichi 贝塞尔/B样条曲线绘制实验
一个基于Taichi框架实现的2D曲线绘制演示程序，核心实现贝塞尔曲线的De Casteljau算法绘制，选做版本扩展了反走样高亮效果、B 样条曲线模式切换功能，通过鼠标交互添加控制点，直观展示参数曲线的生成过程。

## 项目介绍
本项目为计算机图形学实验课项目，利用 Taichi 框架的并行计算能力实现参数曲线的高效绘制：
 1. main.py：实现贝塞尔曲线的 De Casteljau 算法，支持鼠标添加控制点、绿色虚线连接控制点、黄色贝塞尔曲线绘制;
 2. 选做 1（optional_1.py）：优化曲线渲染效果，实现高亮、加粗、反走样的贝塞尔曲线，提升视觉表现;
 3. 选做 2（optional_2.py）：扩展支持贝塞尔 / B样条曲线切换，完善交互逻辑，增加模式指示器。
 4. 全版本支持鼠标交互（左键添加控制点、C键清空），选做2额外支持B键切换曲线模式。

## 技术栈
- Python 3.8+
- Taichi 1.6.0+
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
    └── Work2/
        ├── __init__.py
        ├── README.md     # 项目说明文档
        ├── main.py       # 基础版贝塞尔曲线（主程序）
        ├── optional_1.py # 选做1：高亮反走样贝塞尔曲线
        └── optional_2.py # 选做2：贝塞尔/B样条切换版本
```

## 文件内容
main.py：贝塞尔曲线绘制
核心实现贝塞尔曲线的基础绘制逻辑，包含控制点渲染、绿色虚线连接、黄色贝塞尔曲线生成，是整个项目的核心基础版本。
```
import taichi as ti
import numpy as np
import os

# 强制禁用Vulkan，解决崩溃问题
os.environ["TI_WITH_VULKAN"] = "0"
os.environ["TI_RENDER_BACKEND"] = "cpu"

ti.init(arch=ti.cuda)

W, H = 800, 800
SAMPLING = 1000
MAX_POINTS = 50

# 视觉参数
POINT_RADIUS = 6
LINE_WIDTH  = 2          # 连线粗细
DASH_STEP   = 12         # 虚线步长
DASH_GAP    = 6          # 虚线间隔

# 颜色定义
COLOR_POINT   = (255, 0, 0)         # 控制点：红色
COLOR_LINE    = (0, 255, 0)         # 连线：绿色
COLOR_CURVE   = (255, 255, 0)       # 贝塞尔曲线：黄色

pixels = ti.Vector.field(3, ti.u8, (W, H))
curve_field = ti.Vector.field(2, ti.f32, SAMPLING + 1)

@ti.kernel
def clear():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0, 0, 0], ti.u8)

@ti.kernel
def draw_control_points(points: ti.types.ndarray(), num: ti.i32):
    for k in range(num):
        x, y = points[k, 0], points[k, 1]
        px = int(x * W)
        py = int(y * H)
        for dx in range(-POINT_RADIUS, POINT_RADIUS+1):
            for dy in range(-POINT_RADIUS, POINT_RADIUS+1):
                if dx*dx + dy*dy <= POINT_RADIUS*POINT_RADIUS:
                    if 0 <= px+dx < W and 0 <= py+dy < H:
                        pixels[px+dx, py+dy] = ti.Vector(COLOR_POINT, ti.u8)

@ti.kernel
def draw_green_dashed_lines(points: ti.types.ndarray(), num: ti.i32):
    for k in range(num - 1):
        # 读取控制点坐标并转换为像素坐标
        x0, y0 = points[k, 0], points[k, 1]
        x1, y1 = points[k+1, 0], points[k+1, 1]
        x0_int = int(x0 * W)
        y0_int = int(y0 * H)
        x1_int = int(x1 * W)
        y1_int = int(y1 * H)

        dx = abs(x1_int - x0_int)
        dy = abs(y1_int - y0_int)
        sx = 1 if x0_int < x1_int else -1
        sy = 1 if y0_int < y1_int else -1
        err = dx - dy
        step = 0

        x = ti.cast(x0_int, ti.i32)
        y = ti.cast(y0_int, ti.i32)
        
        while True:
            # 虚线逻辑：一段画，一段空
            if step % (DASH_STEP + DASH_GAP) < DASH_STEP:
                for dw in range(-LINE_WIDTH//2, LINE_WIDTH//2+1):
                    for dh in range(-LINE_WIDTH//2, LINE_WIDTH//2+1):
                        # 计算最终索引并确保是整数
                        px = x + dw
                        py = y + dh
                        if 0 <= px < W and 0 <= py < H:
                            pixels[ti.cast(px, ti.i32), ti.cast(py, ti.i32)] = ti.Vector(COLOR_LINE, ti.u8)
            step += 1

            if x == x1_int and y == y1_int:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

@ti.kernel
def draw_yellow_bezier():
    for i in range(SAMPLING + 1):
        x, y = curve_field[i]
        px = int(x * W)
        py = int(y * H)
        # 黄色实线曲线
        for dw in range(-1, 2):
            for dh in range(-1, 2):
                if 0 <= px+dw < W and 0 <= py+dh < H:
                    pixels[px+dw, py+dh] = ti.Vector(COLOR_CURVE, ti.u8)

def de_casteljau(pts, t):
    p = pts.copy()
    n = len(p)
    for k in range(n-1, 0, -1):
        for i in range(k):
            p[i] = (1-t)*p[i] + t*p[i+1]
    return p[0]

if __name__ == "__main__":
    window = ti.ui.Window("Bezier (Fixed)", (W, H))
    canvas = window.get_canvas()
    controls = []
    last_click = None

    while window.running:
        clear()
        mx, my = window.get_cursor_pos()

        # 鼠标小预览点
        mpx, mpy = int(mx*W), int(my*H)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if 0<=mpx+dx<W and 0<=mpy+dy<H:
                    pixels[mpx+dx, mpy+dy] = ti.Vector(COLOR_POINT, ti.u8)

        # 左键加点
        if window.is_pressed(ti.ui.LMB):
            pos = (mx, my)
            if pos != last_click and len(controls) < MAX_POINTS:
                controls.append(pos)
                last_click = pos
        else:
            last_click = None

        # C清空
        if window.is_pressed('c'):
            controls.clear()

        num = len(controls)
        # 画贝塞尔
        if num >= 2:
            np_pts = np.array(controls, dtype=np.float32)
            curve_np = np.zeros((SAMPLING+1,2), dtype=np.float32)
            for i in range(SAMPLING+1):
                curve_np[i] = de_casteljau(np_pts, i/SAMPLING)
            curve_field.from_numpy(curve_np)
            draw_yellow_bezier()

        # 画控制点 + 绿色虚线
        if num > 0:
            np_controls = np.array(controls, dtype=np.float32)
            draw_control_points(np_controls, num)
            if num >=2:
                draw_green_dashed_lines(np_controls, num)

        canvas.set_image(pixels)
        window.show()
```
        
## 运行方式
```
uv run -m src.Work2.main
```

## 交互说明
- 鼠标左键：在窗口内点击添加红色控制点
- C键（英文输入法）：清空所有控制点
- 鼠标移动：窗口内显示鼠标位置的红色小预览点
- ESC键：关闭窗口

## 演示视频



## 选做内容
### 1.选做1：高亮反走样贝塞尔曲线（optional_1.py）
在基础版基础上优化曲线渲染效果，实现高亮、加粗、反走样的贝塞尔曲线，提升视觉流畅度和美观性。
### 优化点
- 提升采样密度至1500，曲线更连贯
- 6×6超大邻域实现曲线加粗
- 非线性权重计算实现反走样，亮度提升2倍
- 修复索引错误，渲染逻辑更稳定
### 文件内容
```
import taichi as ti
import numpy as np
import os

# 强制禁用Vulkan，永不崩溃
os.environ["TI_WITH_VULKAN"] = "0"
os.environ["TI_RENDER_BACKEND"] = "cpu"

ti.init(arch=ti.cuda)

# 基础配置
W, H = 800, 800
SAMPLING = 1500  # 提高采样密度，曲线更连贯
MAX_POINTS = 50

# 视觉参数
POINT_RADIUS = 6
LINE_WIDTH  = 2
DASH_STEP   = 12
DASH_GAP    = 6

# 颜色
COLOR_POINT   = (255, 0, 0)
COLOR_LINE    = (0, 255, 0)
COLOR_CURVE   = (255, 255, 0)

# 像素场
pixels = ti.Vector.field(3, ti.u8, (W, H))
curve_field = ti.Vector.field(2, ti.f32, SAMPLING + 1)

# 清屏
@ti.kernel
def clear():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0, 0, 0], ti.u8)

# 绘制红色控制点
@ti.kernel
def draw_control_points(points: ti.types.ndarray(), num: ti.i32):
    for k in range(num):
        x, y = points[k, 0], points[k, 1]
        px = int(x * W)
        py = int(y * H)
        for dx in range(-POINT_RADIUS, POINT_RADIUS+1):
            for dy in range(-POINT_RADIUS, POINT_RADIUS+1):
                if dx*dx + dy*dy <= POINT_RADIUS*POINT_RADIUS:
                    if 0 <= px+dx < W and 0 <= py+dy < H:
                        pixels[px+dx, py+dy] = ti.Vector(COLOR_POINT, ti.u8)

# 绘制绿色虚线
@ti.kernel
def draw_green_dashed_lines(points: ti.types.ndarray(), num: ti.i32):
    for k in range(num - 1):
        x0, y0 = points[k, 0], points[k, 1]
        x1, y1 = points[k+1, 0], points[k+1, 1]
        
        x0_i = int(x0 * W)
        y0_i = int(y0 * H)
        x1_i = int(x1 * W)
        y1_i = int(y1 * H)

        dx = abs(x1_i - x0_i)
        dy = abs(y1_i - y0_i)
        sx = 1 if x0_i < x1_i else -1
        sy = 1 if y0_i < y1_i else -1
        err = dx - dy
        step = 0
        x, y = x0_i, y0_i

        while True:
            if step % (DASH_STEP + DASH_GAP) < DASH_STEP:
                for dw in range(-LINE_WIDTH//2, LINE_WIDTH//2+1):
                    for dh in range(-LINE_WIDTH//2, LINE_WIDTH//2+1):
                        cx = x + dw
                        cy = y + dh
                        if 0 <= cx < W and 0 <= cy < H:
                            pixels[cx, cy] = ti.Vector(COLOR_LINE, ti.u8)
            step += 1
            if x == x1_i and y == y1_i: break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

@ti.kernel
def draw_antialiased_bezier():
    brightness = 2.0  # 亮度拉满2倍
    for i in range(SAMPLING + 1):
        fx = curve_field[i][0] * W
        fy = curve_field[i][1] * H
        cx = int(fx)
        cy = int(fy)

        # 6×6超大邻域 → 曲线大幅加粗
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                x = cx + dx
                y = cy + dy
                if 0 <= x < W and 0 <= y < H:
                    # 像素中心
                    pcx = x + 0.5
                    pcy = y + 0.5
                    dist = ti.sqrt((fx - pcx)**2 + (fy - pcy)**2)
                    
                    # 非线性权重：中心极亮，边缘柔和
                    weight = ti.max(0.0, 1.0 - dist / 3.0)
                    weight = weight ** 0.5  # 增强亮度

                    # 颜色计算
                    r = ti.u8(ti.min(255, COLOR_CURVE[0] * weight * brightness))
                    g = ti.u8(ti.min(255, COLOR_CURVE[1] * weight * brightness))
                    b = ti.u8(ti.min(255, COLOR_CURVE[2] * weight * brightness))
                    
                    pixels[x, y] = ti.Vector([r, g, b], ti.u8)

# 贝塞尔算法
def de_casteljau(pts, t):
    p = pts.copy()
    n = len(p)
    for k in range(n-1, 0, -1):
        for i in range(k):
            p[i] = (1-t)*p[i] + t*p[i+1]
    return p[0]
if __name__ == "__main__":
    window = ti.ui.Window("高亮反走样贝塞尔曲线", (W, H))
    canvas = window.get_canvas()
    controls = []
    last_click = None

    while window.running:
        clear()
        mx, my = window.get_cursor_pos()

        # 鼠标预览
        mpx, mpy = int(mx*W), int(my*H)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if 0<=mpx+dx<W and 0<=mpy+dy<H:
                    pixels[mpx+dx, mpy+dy] = ti.Vector(COLOR_POINT, ti.u8)

        # 左键加点
        if window.is_pressed(ti.ui.LMB):
            pos = (mx, my)
            if pos != last_click and len(controls) < MAX_POINTS:
                controls.append(pos)
                last_click = pos
        else:
            last_click = None

        # C清空
        if window.is_pressed('c'):
            controls.clear()

        num = len(controls)
        # 绘制曲线
        if num >= 2:
            np_pts = np.array(controls, dtype=np.float32)
            curve_np = np.zeros((SAMPLING+1,2), dtype=np.float32)
            for i in range(SAMPLING+1):
                curve_np[i] = de_casteljau(np_pts, i/SAMPLING)
            curve_field.from_numpy(curve_np)
            draw_antialiased_bezier()

        # 绘制控制点与虚线
        if num > 0:
            np_controls = np.array(controls, dtype=np.float32)
            draw_control_points(np_controls, num)
            if num >= 2:
                draw_green_dashed_lines(np_controls, num)

        canvas.set_image(pixels)
        window.show()
```
### 运行命令
```
uv run -m src.Work2.optional_1
```
### 演示视频

### 2.选做2：贝塞尔/B样条曲线切换（optional_2.py）
扩展基础版功能，支持贝塞尔曲线与均匀三次 B 样条曲线的切换，增加模式指示器，完善交互逻辑。
### 扩展点
- 实现均匀三次B样条曲线的矩阵计算方法
- B 键切换贝塞尔/B样条模式（左上角颜色指示器：红色=贝塞尔，蓝色=B样条）
- 反走样曲线渲染逻辑复用并优化
- 支持最多100个控制点，采样数提升至1500
### 文件内容
```
import taichi as ti
import numpy as np
import os

os.environ["TI_WITH_VULKAN"] = "0"
os.environ["TI_RENDER_BACKEND"] = "cpu"
ti.init(arch=ti.cuda)

# 基础配置
W, H = 800, 800
MAX_SAMPLING = 1500
MAX_POINTS = 100

POINT_RADIUS = 6
LINE_WIDTH  = 2
DASH_STEP   = 12
DASH_GAP    = 6
BRIGHTNESS  = 2.0
CURVE_RADIUS = 3

COLOR_POINT   = (255, 0, 0)
COLOR_LINE    = (0, 255, 0)
COLOR_CURVE   = (255, 255, 0)
COLOR_BEZIER  = (255, 0, 0)
COLOR_BSPLINE = (0, 0, 255)

pixels = ti.Vector.field(3, ti.u8, (W, H))
curve_points_field = ti.Vector.field(2, ti.f32, MAX_SAMPLING)

@ti.kernel
def clear():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0, 0, 0], ti.u8)

@ti.kernel
def draw_mode_indicator(is_bezier: ti.i32):
    for x in range(10, 30):
        for y in range(10, 30):
            if is_bezier:
                pixels[x, y] = ti.Vector(COLOR_BEZIER, ti.u8)
            else:
                pixels[x, y] = ti.Vector(COLOR_BSPLINE, ti.u8)

# 绘制控制点
@ti.kernel
def draw_control_points(points: ti.types.ndarray(), num: ti.i32):
    for k in range(num):
        x, y = points[k, 0], points[k, 1]
        px = int(x * W)
        py = int(y * H)
        for dx in range(-POINT_RADIUS, POINT_RADIUS+1):
            for dy in range(-POINT_RADIUS, POINT_RADIUS+1):
                if dx*dx + dy*dy <= POINT_RADIUS*POINT_RADIUS:
                    if 0 <= px+dx < W and 0 <= py+dy < H:
                        pixels[px+dx, py+dy] = ti.Vector(COLOR_POINT, ti.u8)

@ti.kernel
def draw_green_dashed_lines(points: ti.types.ndarray(), num: ti.i32):
    for k in range(num - 1):
        x0, y0 = points[k, 0], points[k, 1]
        x1, y1 = points[k+1, 0], points[k+1, 1]
        
        x0_i = int(x0 * W)
        y0_i = int(y0 * H)
        x1_i = int(x1 * W)
        y1_i = int(y1 * H)

        dx = abs(x1_i - x0_i)
        dy = abs(y1_i - y0_i)
        sx = 1 if x0_i < x1_i else -1
        sy = 1 if y0_i < y1_i else -1
        err = dx - dy
        step = 0
        x, y = x0_i, y0_i

        while True:
            if step % (DASH_STEP + DASH_GAP) < DASH_STEP:
                for dw in range(-LINE_WIDTH//2, LINE_WIDTH//2+1):
                    for dh in range(-LINE_WIDTH//2, LINE_WIDTH//2+1):
                        cx = x + dw
                        cy = y + dh
                        if 0 <= cx < W and 0 <= cy < H:
                            pixels[cx, cy] = ti.Vector(COLOR_LINE, ti.u8)
            step += 1
            if x == x1_i and y == y1_i: break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

# 绘制曲线
@ti.kernel
def draw_antialiased_curve(curve_points: ti.template(), total_points: ti.i32):
    for i in range(total_points):
        fx = curve_points[i][0] * W
        fy = curve_points[i][1] * H
        cx = int(fx)
        cy = int(fy)

        for dx in range(-CURVE_RADIUS, CURVE_RADIUS+1):
            for dy in range(-CURVE_RADIUS, CURVE_RADIUS+1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < W and 0 <= y < H:
                    pcx = x + 0.5
                    pcy = y + 0.5
                    dist = ti.sqrt((fx - pcx)**2 + (fy - pcy)**2)
                    weight = ti.max(0.0, 1.0 - dist / (CURVE_RADIUS + 0.5))
                    weight = weight ** 0.5
                    r = ti.u8(ti.min(255, COLOR_CURVE[0] * weight * BRIGHTNESS))
                    g = ti.u8(ti.min(255, COLOR_CURVE[1] * weight * BRIGHTNESS))
                    b = ti.u8(ti.min(255, COLOR_CURVE[2] * weight * BRIGHTNESS))
                    pixels[x, y] = ti.Vector([r, g, b], ti.u8)

# 贝塞尔曲线
def de_casteljau(pts, t):
    p = pts.copy()
    n = len(p)
    for k in range(n-1, 0, -1):
        for i in range(k):
            p[i] = (1-t)*p[i] + t*p[i+1]
    return p[0]

def compute_bezier_points(control_points):
    n = len(control_points)
    if n < 2:
        return np.zeros((0, 2), dtype=np.float32)
    num_samples = MAX_SAMPLING
    curve_points = np.zeros((num_samples, 2), dtype=np.float32)
    for i in range(num_samples):
        t = i / (num_samples - 1)
        curve_points[i] = de_casteljau(control_points, t)
    return curve_points

# B样条曲线
M_BSPLINE = np.array([
    [-1,  3, -3,  1],
    [ 3, -6,  3,  0],
    [-3,  0,  3,  0],
    [ 1,  4,  1,  0]
], dtype=np.float32) / 6.0

def compute_uniform_cubic_bspline_points(control_points):
    n = len(control_points)
    if n < 4:
        return np.zeros((0, 2), dtype=np.float32)
    
    num_segments = n - 3
    samples_per_segment = MAX_SAMPLING // num_segments
    total_samples = num_segments * samples_per_segment
    curve_points = np.zeros((total_samples, 2), dtype=np.float32)

    for seg_idx in range(num_segments):
        p0, p1, p2, p3 = control_points[seg_idx:seg_idx+4]
        P = np.array([p0, p1, p2, p3], dtype=np.float32)
        for sample_idx in range(samples_per_segment):
            t = sample_idx / (samples_per_segment - 1)
            T = np.array([t**3, t**2, t, 1.0], dtype=np.float32)
            pt = T @ M_BSPLINE @ P
            curve_points[seg_idx * samples_per_segment + sample_idx] = pt
    return curve_points

if __name__ == "__main__":
    window = ti.ui.Window("贝塞尔/B样条", res=(W, H))
    canvas = window.get_canvas()
    control_points = []
    is_bezier_mode = True

    while window.running:
        clear()
        mx, my = window.get_cursor_pos()

        # 鼠标预览点
        mpx, mpy = int(mx*W), int(my*H)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if 0<=mpx+dx<W and 0<=mpy+dy<H:
                    pixels[mpx+dx, mpy+dy] = ti.Vector(COLOR_POINT, ti.u8)

        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_POINTS:
                    control_points.append((mx, my))

            if e.key == 'b':
                is_bezier_mode = not is_bezier_mode

            if e.key == 'c':
                control_points.clear()

        # 计算曲线
        num_ctrl = len(control_points)
        curve_points = np.zeros((0, 2), dtype=np.float32)
        if num_ctrl >= 2:
            np_ctrl = np.array(control_points, dtype=np.float32)
            if is_bezier_mode:
                curve_points = compute_bezier_points(np_ctrl)
            else:
                if num_ctrl >= 4:
                    curve_points = compute_uniform_cubic_bspline_points(np_ctrl)
                else:
                    curve_points = compute_bezier_points(np_ctrl)

        # 渲染
        if len(curve_points) > 0:
            curve_points_field.from_numpy(curve_points[:MAX_SAMPLING])
            draw_antialiased_curve(curve_points_field, len(curve_points))

        if num_ctrl > 0:
            np_controls = np.array(control_points, dtype=np.float32)
            draw_control_points(np_controls, num_ctrl)
            if num_ctrl >= 2:
                draw_green_dashed_lines(np_controls, num_ctrl)

        draw_mode_indicator(1 if is_bezier_mode else 0)
        canvas.set_image(pixels)
        window.show()
```
### 运行命令
```
uv run -m src.Work2.optional_2
```
### 演示视频

## 自定义参数
```
W, H = 800, 800           # 窗口大小
SAMPLING = 1000           # 曲线采样精度
MAX_POINTS = 50           # 最大控制点
POINT_RADIUS = 6          # 控制点大小
LINE_WIDTH = 2            # 虚线宽度
COLOR_CURVE = (255,255,0) # 曲线颜色
```

## 常见问题
### 1.运行无反应/报错
- 检查Taichi版本：执行uv add taichi@latest升级到最新版
- 若GPU不支持CUDA，将ti.init(arch=ti.cuda)改为ti.init(arch=ti.cpu)
- 确保输入法为英文状态，避免按键交互失效
### 2.窗口显示异常/曲线不渲染
- 确认控制点数量：基础版最少2个控制点才会生成曲线，B样条模式最少4个
- 检查分辨率参数：确保W/H为正整数，且不超过屏幕分辨率
- 禁用Vulkan：代码已默认设置os.environ["TI_WITH_VULKAN"] = "0"。
### 3.控制点添加失效
- 检查鼠标左键是否正常，避免连续点击同一位置（基础版会过滤重复点击）
- 确认控制点数量未达 MAX_POINTS 上限
