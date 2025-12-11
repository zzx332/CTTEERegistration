import os

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
from png_to_nii import png_to_nii

data_path = r"D:\dataset\nnunet_data\Dataset003\imagesTs"
output_path = r"D:\dataset\nnunet_data\Dataset003\imagesTs"
os.makedirs(output_path, exist_ok=True)
for file in os.listdir(data_path):
    # img = png_to_nii(
    # input_path=os.path.join(data_path, file, f"A2C.png"),
    # output_path=r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images\Patient_0001\A2C.nii.gz",
    # spacing=(1, 1),  # 2D图像只需要(x, y)间距
    # origin=(0.0, 0.0)
    # )
    # img = sitk.ReadImage(os.path.join(data_path, file, f"{file}_4CH_ES_gt.nii.gz"))
    img = sitk.ReadImage(os.path.join(data_path, file))
    # 获取原始尺寸
    original_size = img.GetSize()
    print(f"原始尺寸: {original_size}")
    arr = sitk.GetArrayFromImage(img)
    # arr[arr == 1] = 4
    # 设置目标尺寸为512x512
    # target_size = (645, 504)
    target_size = (800, 600)
    # 计算缩放因子
    zoom_factors = np.array(target_size) / np.array(original_size)

    # 插值（order=1是线性插值，order=0是最近邻，order=3是三次样条）
    arr_resized = zoom(arr, zoom_factors[::-1], order=1)
    img_resized = sitk.GetImageFromArray(arr_resized)
    # img_resized.SetSpacing((1, 1))
    img_resized.SetDirection((-1,0,0,-1))
    sitk.WriteImage(img_resized, os.path.join(output_path, file))