import cv2
import numpy as np
import SimpleITK as sitk
import os


data_path = r"D:\dataset\US\CardiacUDC\cardiacUDC_dataset\label_all_frame"
output_path = r"D:\dataset\nnunet_data\Dataset001\finetune_data"
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
img = sitk.ReadImage(os.path.join(data_path, "patient-67-4_image.nii.gz"))
img_arr = sitk.GetArrayFromImage(img)
label_img = sitk.ReadImage(os.path.join(data_path, "patient-67-4_label.nii.gz"))
label_arr = sitk.GetArrayFromImage(label_img)
for frame in range(label_arr.shape[0]):
    arr = label_arr[frame]
    # 将np.where的结果转换为(x, y)格式的点坐标数组
    def get_points(label_value):
        y_coords, x_coords = np.where(arr == label_value)
        if len(x_coords) == 0:
            return None
        return np.column_stack([x_coords, y_coords]).astype(np.int32)

    points_lv = get_points(1)
    points_rv = get_points(4)
    points_la = get_points(2)
    points_ra = get_points(3)

    # 创建mask
    mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)

    # 填充各个区域
    for points, label in [(points_lv, 1), (points_rv, 4), (points_la, 2), (points_ra, 3)]:
        if points is not None and len(points) > 0:
            cv2.fillPoly(mask, [points], color=label)

    mask_img = sitk.GetImageFromArray(mask)
    mask_img.SetDirection((-1,0,0, -1))
    sitk.WriteImage(mask_img, os.path.join(output_path, "labels", f"patient-67-4_{frame}.nii.gz"))
    img_frame = img_arr[frame]
    img_frame_img = sitk.GetImageFromArray(img_frame)
    img_frame_img.SetDirection((-1,0,0, -1))
    sitk.WriteImage(img_frame_img, os.path.join(output_path,"images", f"patient-67-4_{frame}_0000.nii.gz"))