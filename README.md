# Guideline

规范化2D US-CT配准的workflow

## 预处理

- 使用`totalsegmentator`对CT volume进行分割
```
TotalSegmentator -i D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\CT_resampled_nii\Patient_0000.nii.gz -o D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Segmentation -ta heartchambers_highres --device cpu
```
- 对超声图像进行分割（手动分割或者训练nnunet模型或者尝试阈值分割）
- 使用`us_mask_dim.py`超声分割保存之后矫正维度
## 粗配准（基于四腔质心距离）

- 使用`CT_slice_generate.py`生成CT volume的切片，`center_point_voxel`设置为二尖瓣像素坐标（四腔视图）
- 使用`icp_registration_chamber_pt.py`把单个超声图像和每个切片分别进行配准
- 选取其中结果最好的切片（dice最高或者互信息最好等）

## 精配准（基于四腔边缘的双向chamber距离）

- 把切片的位置映射到原始Volume，平移计算：切片中心物理位置 - CT Volume中心物理位置(可以从切片生成时保存的平移参数文件查询)
- `2D_3D_registration_boundary_gps.py`，`register_icp_multi_label()`。可设置坐标轮换或者是GPS
