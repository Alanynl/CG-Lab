# Taichi 3D 变换演示
一个基于Taichi框架实现的3D坐标变换演示程序，通过键盘交互控制三角形绕Z轴旋转，完整展示模型变换、视图变换与透视投影的经典渲染管线流程。
## 项目介绍
本项目为计算机图形学实验课项目，利用Taichi框架的并行计算能力，实现了从3D顶点到2D屏幕坐标的完整变换：
1. 实现模型变换（绕Z轴旋转）、视图变换、透视投影变换
2. 支持键盘交互（A/D键）控制旋转角度
3. 清晰展示MVP矩阵组合、透视除法与视口变换过程
## 技术栈
- Python 3.8+
- Taichi 1.6.0+
- uv
- Git
## 环境准备
1. 安装 uv
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
2. 配置 uv
```
uv init --python 3.12
uv sync
```
3. 安装 Taichi 包
```
uv add taichi
```
## 项目结构
将代码按以下结构组织：
```
CG-Lab/
│
├── pyproject.toml        # 项目依赖管理文件
└── src/
    └── Work1/
        ├── __init__.py
        ├── README.md     # 项目说明文档
        └── main.py       # 主程序入口
```
## 文件内容
#### main.py
```
import taichi as ti
import math

# 初始化 Taichi，指定使用 CPU 后端
ti.init(arch=ti.cpu)

# 声明 Taichi 的 Field 来存储顶点和转换后的屏幕坐标
vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=3)

@ti.func
def get_model_matrix(angle: float):
    """
    模型变换矩阵：绕 Z 轴旋转
    """
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

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
    # 视线看向 -Z 轴，实际坐标为负
    n = -zNear
    f = -zFar
    
    # 视角转化为弧度并求出 t, b, r, l
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    # 1. 挤压矩阵: 透视平截头体 -> 长方体
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # 2. 正交投影矩阵: 缩放与平移至 [-1, 1]^3
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
    
    # 返回组合矩阵
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle: float):
    """
    在并行架构上计算顶点的坐标变换
    """
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    # MVP 矩阵：右乘原则
    mvp = proj @ view @ model
    
    for i in range(3):
        v = vertices[i]
        # 补全齐次坐标
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法，转化为 NDC 坐标 [-1, 1]
        v_ndc = v_clip / v_clip[3]
        
        # 视口变换：映射到 GUI 的 [0, 1] x [0, 1] 空间
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # 初始化三角形顶点
    vertices[0] = [2.0, 0.0, -2.0]
    vertices[1] = [0.0, 2.0, -2.0]
    vertices[2] = [-2.0, 0.0, -2.0]
    
    # 创建 GUI 窗口
    gui = ti.GUI("3D Transformation (Taichi)", res=(700, 700))
    angle = 0.0
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # 计算变换
        compute_transform(angle)
        
        # 获取 2D 坐标并绘制
        a = screen_coords[0]
        b = screen_coords[1]
        c = screen_coords[2]
        
        gui.line(a, b, radius=2, color=0xFF0000)
        gui.line(b, c, radius=2, color=0x00FF00)
        gui.line(c, a, radius=2, color=0x0000FF)
        
        gui.show()

if __name__ == '__main__':
    main()
```
## 运行程序
```
uv run -m src.Work1.main
```
- 注意：将src目录作为一个模块来执行里面的main文件，输入法处于英文状态
- 运行后会弹出窗口，按A键（逆时针旋转）或D键（顺时针旋转）控制三角形，按ESC退出
## 演示视频
![TRIANGLE](https://github.com/user-attachments/assets/9d74a124-1a7c-4777-a3bc-06c901c3b82c)

## 自定义参数
你可以修改main.py中的参数调整效果：
- 顶点坐标：修改vertices[0]/[1]/[2] 的值改变三角形形状与位置
- 透视参数：修改get_projection_matrix 的 eye_fov（视角）、zNear/zFar（近远平面）
- 旋转步长：修改angle += 10.0 中的数值调整旋转速度
## 常见问题
#### 运行无反应/报错：
- 检查 Taichi 版本：uv add taichi@latest 升级到最新版
- 若 GPU 不支持，将 ti.init(arch=ti.cpu) 保持 CPU 模式（代码默认 CPU）
#### 窗口显示异常：
- 确认顶点坐标在视锥体范围内（z值在zNear和zFar之间）
- 检查 aspect_ratio 与窗口分辨率比例是否匹配
