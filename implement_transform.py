import SimpleITK as sitk

fixed_img = sitk.ReadImage(r"D:\dataset\TEECT_data\ct\ct_train_1004_image\slice_121_t5.0_rx50_ry-25.nii.gz")
moving_img = sitk.ReadImage(r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_label.nii.gz")
moving_img = sitk.Flip(moving_img, flipAxes=[True, True])  # X轴和Y轴都翻转
moving_img.SetOrigin((0,0))
transform = sitk.ReadTransform(r"D:\dataset\TEECT_data\registration_results\icp_chamber_pt\slice_060_label.nii_transform.tfm")
# inverse_transform = transform.GetInverse()
# resampled_img = sitk.Resample(moving_img, fixed_img, transform, sitk.sitkLinear, 0.0)
resampled_img = sitk.Resample(moving_img, fixed_img, transform, sitk.sitkNearestNeighbor, 0.0)

sitk.WriteImage(resampled_img, r"D:\dataset\TEECT_data\registration_results\icp_chamber_pt\slice_060_label_initial_transform.nii.gz")