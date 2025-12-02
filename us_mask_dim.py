import numpy as np
import os
import SimpleITK as sitk

label_path = r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images\Patient_0036"
# img = sitk.ReadImage(os.path.join(label_path, "A4C_seg.nii.gz"))
img = sitk.ReadImage(os.path.join(label_path, "A4C.nii.gz"))
arr = sitk.GetArrayFromImage(img)
arr = arr.squeeze()
img_1 = sitk.GetImageFromArray(arr)
sitk.WriteImage(img_1, os.path.join(label_path, "A4C1.nii.gz"))
# print(np.unique(arr))