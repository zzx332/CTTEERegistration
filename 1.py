import SimpleITK as sitk
import numpy as np
from pathlib import Path

input_path = r"D:\dataset\CT\MM-WHS2017\ct_train\ct_train_1004_label.nii"

chamber_to_edge_map = {
    420: 2,  # LV (左心房) -> 1
    500: 1,  # LA (左心室) -> 2
    600: 4,  # RV (右心室) -> 4
    550: 3,  # RA (右心房) -> 3
}
seg_img = sitk.ReadImage(input_path)
seg_array = sitk.GetArrayFromImage(seg_img)

print(f"处理: {Path(input_path).name}")
print(f"  原始标签: {np.unique(seg_array)}")

# 创建新的数组（初始化为0）
new_array = np.zeros_like(seg_array, dtype=np.uint8)

# 应用映射
for old_label, new_label in chamber_to_edge_map.items():
    mask = (seg_array == old_label)
    new_array[mask] = new_label
    pixel_count = np.sum(mask)
    if pixel_count > 0:
        print(f"  {old_label} -> {new_label}: {pixel_count} 像素")

print(f"  映射后标签: {np.unique(new_array)}")

# 转换回SimpleITK图像
new_img = sitk.GetImageFromArray(new_array)
new_img.CopyInformation(seg_img)

# 保存
output_path = r"D:\dataset\CT\MM-WHS2017\ct_train\ct_train_1004_remapped_label.nii"

sitk.WriteImage(new_img, str(output_path), useCompression=True)
print(f"  ✓ 已保存: {output_path}\n")