import os

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom

data_path = r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images\Patient_0001"
output_path = r"D:\dataset\nnunet_data\Dataset001\imagesTs"

img = sitk.ReadImage(os.path.join(data_path, "A4C_0000.nii.gz"))
# 获取原始尺寸
original_size = img.GetSize()
print(f"原始尺寸: {original_size}")
arr = sitk.GetArrayFromImage(img)
# 设置目标尺寸为512x512
target_size = (645, 504)
# 计算缩放因子
zoom_factors = np.array(target_size) / np.array(original_size)

# 插值（order=1是线性插值，order=0是最近邻，order=3是三次样条）
arr_resized = zoom(arr, zoom_factors[::-1], order=1)
img_resized = sitk.GetImageFromArray(arr_resized)
img_resized.SetSpacing((1, 1))
sitk.WriteImage(img_resized, os.path.join(output_path, "A4C_0000.nii.gz"))