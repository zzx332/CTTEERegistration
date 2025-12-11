import SimpleITK as sitk
import numpy as np
import os

data_path = r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images\Patient_0000"
img = sitk.ReadImage(os.path.join(data_path, "A2C_seg.nii.gz"))
arr = sitk.GetArrayFromImage(img)
# arr = arr[0,:,:,0]  # image
# arr = arr.squeeze()  # label
# arr[arr == 1] = 4
# arr[arr == 2] = 3
img_1 = sitk.GetImageFromArray(arr)
sitk.WriteImage(img_1, os.path.join(data_path, "A2C_seg.nii.gz"))