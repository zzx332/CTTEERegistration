import SimpleITK as sitk

# ct = sitk.ReadImage(r"D:\dataset\TEECT_data\ct\ct_train_1004_label\slice_121_t5.0_rx50_ry-25.nii.gz")
ct = sitk.ReadImage(r"D:\dataset\TEECT_data\ct\label.nii.gz")
# ct = sitk.ReadImage(r"D:\dataset\TEECT_data\registration_results\2d_3d\extracted_2d.nii.gz")
label = sitk.ReadImage(r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_label_initial_transform.nii.gz")

print("CT Direction:", ct.GetDirection())
print("Label Direction:", label.GetDirection())
print("CT Origin:", ct.GetOrigin())
print("Label Origin:", label.GetOrigin())
print("CT Spacing:", ct.GetSpacing())
print("Label Spacing:", label.GetSpacing())

# 检查某个像素坐标是否映射到相同物理位置
pixel_idx = [100, 100]
ct_physical = ct.TransformIndexToPhysicalPoint(pixel_idx)
label_physical = label.TransformIndexToPhysicalPoint(pixel_idx)

print(f"CT像素{pixel_idx}的物理坐标: {ct_physical}")
print(f"Label像素{pixel_idx}的物理坐标: {label_physical}")