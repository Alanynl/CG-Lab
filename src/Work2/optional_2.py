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