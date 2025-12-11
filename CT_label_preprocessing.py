import SimpleITK as sitk
import os
import numpy as np

ct_path = r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\CT_resampled_nii"
for patient_file in os.listdir(ct_path):
    patient_id = patient_file.split(".")[0]
    if patient_id != "Patient_0006":
        continue
    label_path = os.path.join(r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Segmentation", patient_id)

    ct = sitk.ReadImage(os.path.join(ct_path, patient_file))
    label_3 = sitk.ReadImage(os.path.join(label_path, "heart_atrium_left.nii.gz"))
    label_2 = sitk.ReadImage(os.path.join(label_path, "heart_atrium_right.nii.gz"))
    label_4 = sitk.ReadImage(os.path.join(label_path, "heart_ventricle_left.nii.gz"))
    label_1 = sitk.ReadImage(os.path.join(label_path, "heart_ventricle_right.nii.gz"))


    arr_4 = sitk.GetArrayFromImage(label_4)
    arr_1 = sitk.GetArrayFromImage(label_1)
    arr_3 = sitk.GetArrayFromImage(label_3)
    arr_2 = sitk.GetArrayFromImage(label_2)
    arr_4 = arr_4.squeeze()
    arr_1 = arr_1.squeeze()
    arr_3 = arr_3.squeeze()
    arr_2 = arr_2.squeeze()

    label_array = np.zeros_like(arr_4, dtype=np.uint8)
    label_array[arr_4 != 0] = 1
    label_array[arr_1 != 0] = 4
    label_array[arr_3 != 0] = 2
    label_array[arr_2 != 0] = 3
    label_array = sitk.GetImageFromArray(label_array)
    label_array.CopyInformation(ct)
    sitk.WriteImage(label_array, os.path.join(label_path, f"{patient_id}_label.nii.gz"))
    print(f"Processed {patient_id}")