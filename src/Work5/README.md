- 姓名：陈蓝薪
- 学号：202411998405
- 专业：人工智能
# 奶牛模型可微渲染剪影优化实验
基于PyTorch3D实现的奶牛3D模型剪影驱动形变优化系统，通过多视角剪影损失约束，实现从初始球体到奶牛模型的高精度几何重建，支持6视角（正背+4斜向）联合优化，适配CUDA加速计算。

## 项目介绍
本项目为计算机图形学可微渲染与3D重建方向实验课项目，利用PyTorch3D的可微渲染管线与自动微分机制，实现基于多视角剪影的3D模型形变优化：
- main.py：核心版剪影优化实现，完成奶牛模型解析加载、6视角相机系统构建、可微渲染管线搭建、多损失函数联合优化、训练过程可视化与结果保存。

## 技术栈
- Python 3.12+
- PyTorch 2.2+（自动微分/优化器）
- PyTorch3D 0.7.4（可微渲染/3D几何处理）
- NumPy（数值计算）
- Matplotlib（结果可视化）
- Git（版本控制）
- OBJ模型解析库（PyTorch3D内置）

## 环境准备
### 安装配置
```
# 创建并激活虚拟环境
conda create -n pytorch3d_env python=3.12
conda activate pytorch3d_env

# 安装PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 安装其他依赖
pip install numpy matplotlib pathlib
```

## 项目结构
```
CG-Lab/
│
├── pyproject.toml        # 项目依赖管理文件
└── src/
    └── Work5/
        ├── __init__.py
        ├── README.md     # 项目说明文档
        ├── main.py       # 核心版：奶牛模型剪影优化+多视角渲染+训练可视化
        └── cow.obj       # 奶牛3D模型OBJ文件
```

## 文件内容
### main.py：奶牛模型剪影驱动形变优化核心实现
核心功能：从初始球体出发，通过6视角剪影损失约束，联合拉普拉斯平滑、边长度、法向量一致性等几何先验，实现奶牛模型的高精度重建。

```
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams
)
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
    mesh_edge_loss,
    mesh_normal_consistency,
)

# 本地奶牛模型路径
def get_cow_mesh(device):
    obj_path = r"C:\trae_projects\CG - Lab\src\Work5\cow.obj"
    obj_path = Path(obj_path)
    if not obj_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{obj_path}")
    verts, faces, _ = load_obj(str(obj_path))
    verts = verts.to(device)
    faces = faces.verts_idx.to(device)
    verts = verts - verts.mean(0)
    verts = verts / verts.abs().max().item()
    return Meshes(verts=[verts], faces=[faces]).to(device)

# 超参微调
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
NUM_ITER = 3000
LR = 1e-4
LEVEL = 3
# 6视角：前后+4个斜角
NUM_VIEWS = 6
# azim：0,180(正背)+45,135,225,315(四个斜视角)
az_list = [0, 180, 45, 135, 225, 315]

W_SIL = 2.8    # 小幅提高剪影权重，强制收边
W_LAP = 0.03
W_NORMAL = 0.02
W_EDGE = 0.005

# 相机
target_mesh = get_cow_mesh(device)
R, T = look_at_view_transform(dist=2.5, elev=0, azim=az_list)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
target_mesh = target_mesh.extend(NUM_VIEWS)

# 光栅
blend_params = BlendParams(sigma=3e-4, gamma=3e-4)
raster_settings = RasterizationSettings(
    image_size=IMAGE_SIZE,
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
    faces_per_pixel=40,
)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

# 初始化球体
src_mesh = ico_sphere(level=LEVEL, device=device).extend(NUM_VIEWS)
deform_verts = nn.Parameter(torch.zeros_like(src_mesh.verts_packed()), requires_grad=True)
optimizer = optim.Adam([deform_verts], lr=LR)

# 预渲染剪影
with torch.no_grad():
    target_sil = renderer(target_mesh)[..., 3]
    target_front_view = target_sil[1].cpu().numpy()
    target_back_view = target_sil[0].cpu().numpy()

loss_history = []
print(f"开始优化：{NUM_ITER}轮 | 6视角(正背+4斜向) | 设备：{device}")
print(f"模型路径：C:\\trae_projects\\CG - Lab\\src\\Work5\\cow.obj")

# 训练循环
for i in range(NUM_ITER):
    optimizer.zero_grad()
    new_mesh = src_mesh.offset_verts(deform_verts)
    pred_sil = renderer(new_mesh)[..., 3]

    loss_sil = ((pred_sil - target_sil) ** 2).mean()
    loss_lap = mesh_laplacian_smoothing(new_mesh)
    loss_edge = mesh_edge_loss(new_mesh)
    loss_normal = mesh_normal_consistency(new_mesh)

    total_loss = W_SIL * loss_sil + W_LAP * loss_lap + W_EDGE * loss_edge + W_NORMAL * loss_normal
    total_loss.backward()
    optimizer.step()
    loss_history.append(total_loss.item())

    if i % 300 == 0:
        print(f"Iter {i:04d} | Total Loss: {total_loss.item():.6f} | 剪影Loss: {loss_sil.item():.6f}")
        with torch.no_grad():
            pred_front = pred_sil[1].cpu().numpy()
            pred_back = pred_sil[0].cpu().numpy()
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'迭代 {i} | 6斜视角优化', fontsize=16)
        axs[0, 0].imshow(target_front_view, cmap='gray');axs[0,0].set_title('目标（正面）');axs[0,0].axis('off')
        axs[0, 1].imshow(pred_front, cmap='gray');axs[0,1].set_title(f'预测（正面）');axs[0,1].axis('off')
        axs[1, 0].imshow(target_back_view, cmap='gray');axs[1,0].set_title('目标（背面）');axs[1,0].axis('off')
        axs[1, 1].imshow(pred_back, cmap='gray');axs[1,1].set_title(f'预测（背面）');axs[1,1].axis('off')
        plt.tight_layout()
        plt.show()

# 收尾绘图
print("✅ 优化完成！")
plt.figure(figsize=(10, 4))
plt.plot(loss_history, label='总损失')
plt.xlabel('迭代次数');plt.ylabel('损失值')
plt.title('训练损失曲线（6视角优化）');plt.grid(True, alpha=0.3);plt.legend()
plt.show()

# 最终对比
with torch.no_grad():
    final_mesh = src_mesh.offset_verts(deform_verts)
    final_sil = renderer(final_mesh)[..., 3]
    final_front = final_sil[1].cpu().numpy()
    final_back = final_sil[0].cpu().numpy()
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('最终结果对比', fontsize=16)
axs[0,0].imshow(target_front_view,cmap='gray');axs[0,0].set_title('目标（正面）');axs[0,0].axis('off')
axs[0,1].imshow(final_front,cmap='gray');axs[0,1].set_title('最终预测（正面）');axs[0,1].axis('off')
axs[1,0].imshow(target_back_view,cmap='gray');axs[1,0].set_title('目标（背面）');axs[1,0].axis('off')
axs[1,1].imshow(final_back,cmap='gray');axs[1,1].set_title('最终预测（背面）');axs[1,1].axis('off')
plt.tight_layout();plt.show()

# 保存权重
os.makedirs('output',exist_ok=True)
torch.save({'deform_verts': deform_verts,'loss_history': loss_history}, 'output/cow_deformation_final.pth')
print("📁 结果已保存至 output 文件夹")
```

## 运行方式
```
# 激活环境
conda activate pytorch3d_env

# 运行主程序
python src/Work5/main.py
```

## 结果展示
<img width="1174" height="1249" alt="优化（1500）" src="https://github.com/user-attachments/assets/699c879e-b185-4931-8485-edc1348c0afd" />
<img width="1224" height="1116" alt="优化结果对比" src="https://github.com/user-attachments/assets/1ded3c75-adfc-4bbb-9101-bdcd1a76497e" />
<img width="1500" height="687" alt="优化损失函数" src="https://github.com/user-attachments/assets/6429ca3f-2f5b-476b-accf-908dbe2f133f" />
<img width="679" height="388" alt="输出" src="https://github.com/user-attachments/assets/d7c4e798-6500-4cea-808d-42577566edbc" />


## 交互说明
- 自动训练过程：程序启动后自动执行3000轮优化，每300轮显示一次正面/背面剪影对比
- 可视化输出：
  - 训练过程：实时显示目标与预测剪影对比，便于观察优化进展
  - 训练结束：显示完整损失曲线与最终结果对比图
- 结果保存：自动将最终形变顶点与损失历史保存至`output`文件夹
- ESC键：关闭所有Matplotlib窗口，终止程序运行

## 自定义参数
```
# 核心超参（推荐调整范围）
IMAGE_SIZE = 256         # 渲染分辨率（256/512，更高更清晰但更慢）
NUM_ITER = 3000          # 迭代次数（2000-5000，更多迭代更精细）
LR = 1e-4                # 学习率（5e-5~2e-4，过大易震荡，过小收敛慢）
LEVEL = 3                # 初始球体细分级别（2-4，级别越高顶点越多）
NUM_VIEWS = 6            # 视角数量（2-8，更多视角约束更强）
az_list = [0, 180, 45, 135, 225, 315]  # 视角方位角列表

# 损失权重（关键调参）
W_SIL = 2.8    # 剪影损失权重（2.0-4.0，越大轮廓越贴合但易有噪点）
W_LAP = 0.03   # 拉普拉斯平滑（0.01-0.1，越大越平滑但细节越少）
W_NORMAL = 0.02 # 法向量一致性（0.01-0.05，增强表面连续性）
W_EDGE = 0.005 # 边长度损失（0.001-0.01，防止局部过度拉伸）
```
## 运行结果展示

## 常见问题
### 1. PyTorch3D安装失败
- CUDA版本不匹配：确保PyTorch与PyTorch3D的CUDA版本一致（推荐11.8）
- Windows系统问题：优先使用conda安装，或从[PyTorch3D官方文档](https://pytorch3d.org/docs/installation)获取Windows安装指南
- 版本冲突：创建全新虚拟环境，避免与其他库冲突

### 2. 模型加载失败/无显示
- 路径错误：检查`obj_path`是否正确指向`cow.obj`文件，建议使用绝对路径
- 模型格式问题：确保OBJ文件仅包含三角面片，无复杂材质定义
- 顶点范围异常：代码已包含中心归一化和缩放，若仍异常可手动调整缩放系数

### 3. 侧边凸起/轮廓不贴合
- 增加剪影权重：将`W_SIL`提高至3.0-3.5，强制轮廓向内收缩
- 调整视角：添加纯左右正交视角（azim=90,270），增强侧边约束
- 续训优化：加载已有权重继续训练1000轮，学习率降至5e-5

### 4. 训练速度慢/GPU占用高
- 降低分辨率：将`IMAGE_SIZE`改为128，减少计算量
- 减少视角：临时将`NUM_VIEWS`改为2，快速验证效果
- 降低细分级别：将`LEVEL`改为2，减少顶点数量

## 实验结果说明
1. 收敛表现：损失从初始0.6+平稳下降至0.09+，剪影损失同步下降，无震荡发散
2. 关键改进：
   - 正反面标签修正完毕，与目标完全对应
   - 6视角方案有效抑制球形鼓包，整体外形贴合奶牛轮廓
   - 头顶犄角、双腿缝隙等关键部位成型良好
3. 现存局限：侧边零星小凸起属于剪影重建天然局限，无纯左右正交视角时难以完全消除，可通过微调超参进一步优化
