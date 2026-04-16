import taichi as ti
import math

ti.init(arch=ti.cpu)

NUM_VERTICES = 8
NUM_EDGES = 12

vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)

# 定义立方体12条边的顶点索引对，每条边用 (顶点1索引, 顶点2索引) 表示
edges = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # 前面（z=1）
    (4, 5), (5, 7), (7, 6), (6, 4),  # 后面（z=-1）
    (0, 4), (1, 5), (2, 6), (3, 7)   # 连接前后的边
]

@ti.func
def get_model_matrix(angle_y: float, angle_z: float):
    """
    模型变换矩阵：绕 Y 轴和 Z 轴旋转（增强3D效果）
    """
    # 绕Y轴旋转矩阵
    rad_y = angle_y * math.pi / 180.0
    c_y = ti.cos(rad_y)
    s_y = ti.sin(rad_y)
    rot_y = ti.Matrix([
        [c_y,  0.0, s_y, 0.0],
        [0.0,  1.0, 0.0, 0.0],
        [-s_y, 0.0, c_y, 0.0],
        [0.0,  0.0, 0.0, 1.0]
    ])
    
    # 绕Z轴旋转矩阵
    rad_z = angle_z * math.pi / 180.0
    c_z = ti.cos(rad_z)
    s_z = ti.sin(rad_z)
    rot_z = ti.Matrix([
        [c_z, -s_z, 0.0, 0.0],
        [s_z,  c_z, 0.0, 0.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0,  0.0, 0.0, 1.0]
    ])
    
    # 先绕Y轴，再绕Z轴
    return rot_z @ rot_y

@ti.func
def get_view_matrix(eye_pos):
    """
    视图变换矩阵：将相机移动到原点
    """
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: float, aspect_ratio: float, zNear: float, zFar: float):
    """
    透视投影矩阵
    """
    # 视线看向-Z轴，实际坐标为负
    n = -zNear
    f = -zFar
    
    # 视角转化为弧度并求出 t, b, r, l
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    # 挤压矩阵: 透视平截头体 -> 长方体
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # 正交投影矩阵: 缩放与平移至 [-1, 1]^3
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho = M_ortho_scale @ M_ortho_trans
    
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle_y: float, angle_z: float):
    """
    在并行架构上计算顶点的坐标变换
    """
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle_y, angle_z)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    # MVP矩阵
    mvp = proj @ view @ model
    
    # 遍历8个顶点进行变换
    for i in range(NUM_VERTICES):
        v = vertices[i]
        # 补全齐次坐标
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法，转化为NDC坐标
        v_ndc = v_clip / v_clip[3]
        
        # 视口变换：映射到 GUI 的 [0, 1] x [0, 1] 空间
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # 初始化立方体的8个顶点
    # 顶点顺序：前面(z=1)的四个点 -> 后面(z=-1)的四个点
    vertices[0] = [1.0, 1.0, 1.0]   # 前右上
    vertices[1] = [1.0, 1.0, -1.0]  # 前左上
    vertices[2] = [1.0, -1.0, 1.0]  # 前右下
    vertices[3] = [1.0, -1.0, -1.0] # 前左下
    vertices[4] = [-1.0, 1.0, 1.0]  # 后右上
    vertices[5] = [-1.0, 1.0, -1.0] # 后左上
    vertices[6] = [-1.0, -1.0, 1.0] # 后右下
    vertices[7] = [-1.0, -1.0, -1.0]# 后左下
    
    # 创建GUI窗口
    gui = ti.GUI("3D Cube with Perspective (Taichi)", res=(700, 700))
    angle_y = 0.0  # 绕Y轴旋转角度
    angle_z = 0.0  # 绕Z轴旋转角度
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle_y += 10.0  # 按A键绕Y轴逆时针转
            elif gui.event.key == 'd':
                angle_y -= 10.0  # 按D键绕Y轴顺时针转
            elif gui.event.key == 'w':
                angle_z += 10.0  # 按W键绕Z轴逆时针转
            elif gui.event.key == 's':
                angle_z -= 10.0  # 按S键绕Z轴顺时针转
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        # 计算变换
        compute_transform(angle_y, angle_z)
        
        # 遍历12条边进行绘制
        for v1, v2 in edges:
            a = screen_coords[v1]
            b = screen_coords[v2]
            gui.line(a, b, radius=2, color=0x00BFFF)  # 天蓝色线框
        
        gui.show()

if __name__ == '__main__':
    main()
