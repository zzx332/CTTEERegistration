import SimpleITK as sitk
import numpy as np
from pathlib import Path
from scipy import ndimage

# 输入路径
img_path = r"D:\dataset\US\CardiacUDC\cardiacUDC_dataset\Site_G_20\patient-1-4_image.nii"
label_path = r"D:\dataset\US\CardiacUDC\cardiacUDC_dataset\Site_G_20\patient-1-4_label.nii"

# 输出目录
output_dir = Path(r"D:\dataset\TEECT_data\tee\patient-1-4")
output_dir.mkdir(parents=True, exist_ok=True)

# 读取图像和标签
img = sitk.ReadImage(img_path)
img_array = sitk.GetArrayFromImage(img)  # shape: (Z, Y, X)
label = sitk.ReadImage(label_path)
label_array = sitk.GetArrayFromImage(label)

print(f"图像形状: {img_array.shape}")
print(f"标签形状: {label_array.shape}")
print(f"图像spacing: {img.GetSpacing()}")
print(f"图像origin: {img.GetOrigin()}")

# 获取图像元数据
# spacing = img.GetSpacing()
spacing = img.GetSpacing()
origin = img.GetOrigin()
direction = img.GetDirection()

# 遍历所有切片，保存label不为0的切片
saved_count = 0
total_slices = img_array.shape[0]

print(f"\n开始处理 {total_slices} 个切片...")

for slice_idx in range(total_slices):
    # 获取当前切片的label
    label_slice = label_array[slice_idx]
    
    # 检查是否包含非0标签
    if np.any(label_slice != 0):
        non_zero_count = np.sum(label_slice != 0)
        if non_zero_count < 5000:
            continue
        # 获取对应的图像切片
        img_slice = img_array[slice_idx]

        # 去除图像中的EGG信号
        mask_182 = (img_slice == 182)

        if np.any(mask_182):
            # 标记连通域
            # 定义8邻域结构
            structure_8 = ndimage.generate_binary_structure(2, 2)  # 2D图像，2表示8邻域
            labeled_array, num_features = ndimage.label(mask_182, structure=structure_8)
            if num_features > 0:
                # 找到最大连通域
                sizes = ndimage.sum(mask_182, labeled_array, range(1, num_features + 1))
                max_label = np.argmax(sizes) + 1
                # 只移除最大连通域
                img_slice[labeled_array == max_label] = 0
        
        # 转换为SimpleITK图像对象（2D）
        img_slice_sitk = sitk.GetImageFromArray(img_slice)
        label_slice_sitk = sitk.GetImageFromArray(label_slice.astype(np.uint8))

        
        # 计算2D切片的origin（考虑z方向的偏移）
        slice_origin = list(origin)
        slice_origin[2] = origin[2] + slice_idx * spacing[2]
        img_slice_sitk.SetOrigin([slice_origin[0], slice_origin[1]])
        label_slice_sitk.SetOrigin([slice_origin[0], slice_origin[1]])
        
        # 设置2D切片的spacing（只取x和y）
        img_slice_sitk.SetSpacing([0.3, 0.3])
        label_slice_sitk.SetSpacing([0.3, 0.3])
        # 设置2D方向矩阵（3x3 -> 2x2）
        direction_3d = np.array(direction).reshape(3, 3)
        direction_2d = direction_3d[:2, :2].flatten().tolist()
        img_slice_sitk.SetDirection(direction_2d)
        label_slice_sitk.SetDirection(direction_2d)
        
        # 保存切片
        img_output_path = output_dir / f"slice_{slice_idx:03d}_image.nii.gz"
        label_output_path = output_dir / f"slice_{slice_idx:03d}_label.nii.gz"
        
        sitk.WriteImage(img_slice_sitk, str(img_output_path), useCompression=True)
        sitk.WriteImage(label_slice_sitk, str(label_output_path), useCompression=True)
        
        saved_count += 1
        
        # 统计标签信息
        unique_labels = np.unique(label_slice)

        
        print(f"  保存切片 {slice_idx:03d}: 标签值={unique_labels}, 非零像素数={non_zero_count}")

print(f"\n完成！共保存 {saved_count}/{total_slices} 个包含标注的切片")
print(f"保存位置: {output_dir}")