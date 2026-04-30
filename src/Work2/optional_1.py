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