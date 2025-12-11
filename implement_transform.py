import SimpleITK as sitk

fixed_img = sitk.ReadImage(r"D:\dataset\TEECT_data\ct_paired\Patient_0036_image\slice_020_t-5.0_rx50_ry-50.nii.gz")
# moving_img = sitk.ReadImage(r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images\Patient_0006\PSAX_PAP_masked.nii.gz")
moving_img = sitk.ReadImage(r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images\Patient_0036\A2C_seg.nii.gz")
# moving_img = sitk.Flip(moving_img, flipAxes=[True, True])  # X轴和Y轴都翻转
# moving_img.SetOrigin((0,0))
transform = sitk.ReadTransform(r"D:\dataset\TEECT_data\registration_results_paired\icp_chamber_pt_batch\Patient_0036_A2C\slice_020_t-5.0_rx50_ry-50.nii_transform.tfm")
inverse_transform = transform.GetInverse()
# resampled_img = sitk.Resample(moving_img, fixed_img, inverse_transform, sitk.sitkLinear, 0.0)
resampled_img = sitk.Resample(moving_img, fixed_img, inverse_transform, sitk.sitkNearestNeighbor, 0.0)

sitk.WriteImage(resampled_img, r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images\Patient_0036\A2C_seg_initial_transform.nii.gz")