import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

"""
可视化配准结果
"""
ultrasound_2d_img = sitk.ReadImage(r"D:\dataset\TEECT_data\registration_results\icp_chamber_pt\slice_060_image_initial_transform.nii.gz")
ct_img = sitk.ReadImage(r"D:\dataset\TEECT_data\ct\ct_train_1004_image\slice_121_t5.0_rx50_ry-25.nii.gz")
# 获取图像
us_array = sitk.GetArrayFromImage(ultrasound_2d_img).squeeze()
slice_array = sitk.GetArrayFromImage(ct_img).squeeze()

# 归一化
def normalize(img):
    p1, p99 = np.percentile(img[img > -500], [1, 99])
    return np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)

us_norm = normalize(us_array)
slice_norm = normalize(slice_array)

fig = plt.figure(figsize=(18, 6))

# 1. 超声图像
ax1 = plt.subplot(1, 3, 1)
ax1.imshow(us_norm, cmap='gray')
ax1.set_title('2D Ultrasound', fontsize=14)
ax1.axis('off')

# 2. 提取的CT切片
ax2 = plt.subplot(1, 3, 2)
ax2.imshow(slice_norm, cmap='gray')
ax2.set_title('Extracted CT Slice', fontsize=14)
ax2.axis('off')

# 3. 叠加图
ax3 = plt.subplot(1, 3, 3)
overlay = np.zeros((*us_norm.shape, 3))
overlay[:, :, 0] = us_norm  # 红色：超声
overlay[:, :, 1] = slice_norm  # 绿色：CT切片
ax3.imshow(overlay)
ax3.set_title('Overlay (Red=US, Green=CT)', fontsize=14)
ax3.axis('off')

plt.tight_layout()

# 保存
vis_path = r"D:\dataset\TEECT_data\registration_results\icp_chamber_pt\registration_result.png" 
plt.savefig(vis_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ 可视化已保存: {vis_path}")