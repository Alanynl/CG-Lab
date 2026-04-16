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