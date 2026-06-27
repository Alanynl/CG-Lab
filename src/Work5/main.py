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