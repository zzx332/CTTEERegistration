import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt


class CT3DSliceGenerator:
    """
    从3D CT图像中生成多个候选2D切片的生成器
    
    功能：
    - 支持法向平移（±5mm，步长2.5mm）
    - 支持双轴旋转（±90度，步长45度）
    - 生成5×5×5=125个候选平面
    """
    
    def __init__(self, ct_image_path: str):
        """
        初始化CT切片生成器
        
        Args:
            ct_image_path: 3D CT图像路径（支持DICOM系列或NIfTI格式）
        """
        self.image = self._load_image(ct_image_path)
        self.spacing = np.array(self.image.GetSpacing())
        self.origin = np.array(self.image.GetOrigin())
        self.size = np.array(self.image.GetSize())
        self.direction = np.array(self.image.GetDirection()).reshape(3, 3)    
        self.ct_center = self.origin + self.direction @ (self.size * self.spacing / 2.0)
        
    def _load_image(self, image_path: str) -> sitk.Image:
        """
        加载CT图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            SimpleITK图像对象
        """
        path = Path(image_path)
        
        if path.is_dir():
            # 加载DICOM系列
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(path))
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            # 加载单个文件（NIfTI等）
            image = sitk.ReadImage(str(path))
            
        return image
    
    def voxel_to_physical(self, voxel_coord: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        将体素坐标转换为物理坐标（考虑图像方向矩阵）
        
        正确的转换公式：physical = origin + direction @ (voxel * spacing)
        
        Args:
            voxel_coord: 体素坐标 (i, j, k)
            
        Returns:
            物理坐标 (x, y, z) 单位：mm
        """
        voxel = np.array(voxel_coord)
        
        # 获取方向矩阵（3x3）- Direction定义了图像坐标系的方向
        direction = np.array(self.image.GetDirection()).reshape(3, 3)
        
        # 正确的转换公式：考虑origin、spacing和direction
        physical = self.origin + direction @ (voxel * self.spacing)
        
        return tuple(physical)
    
    def physical_to_voxel(self, physical_coord: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        将物理坐标转换为体素坐标（考虑图像方向矩阵）
        
        正确的转换公式：voxel = (direction^-1 @ (physical - origin)) / spacing
        
        Args:
            physical_coord: 物理坐标 (x, y, z) 单位：mm
            
        Returns:
            体素坐标 (i, j, k)
        """
        physical = np.array(physical_coord)
        
        # 获取方向矩阵（3x3）
        direction = np.array(self.image.GetDirection()).reshape(3, 3)
        
        # 逆向转换：先减去原点，用逆方向矩阵变换，再除以间距
        voxel = (np.linalg.inv(direction) @ (physical - self.origin)) / self.spacing
        
        return tuple(voxel)
    
    def _create_rotation_matrix(self, axis: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        创建绕指定轴旋转的旋转矩阵（Rodrigues公式）
        
        Args:
            axis: 旋转轴（单位向量）
            angle_deg: 旋转角度（度）
            
        Returns:
            3×3旋转矩阵
        """
        angle_rad = np.radians(angle_deg)
        axis = axis / np.linalg.norm(axis)
        
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        
        # Rodrigues旋转公式
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
        return R
    
    def generate_slice_planes(
        self,
        center_point: Tuple[float, float, float],
        normal_vector: Tuple[float, float, float],
        translation_range: float = 5.0,
        translation_step: float = 2.5,
        rotation_range: float = 90.0,
        rotation_step: float = 45.0,
        slice_size: Tuple[int, int] = (256, 256),
        slice_spacing: Tuple[float, float] = (1.0, 1.0),
        use_voxel_coord: bool = False
    ) -> List[dict]:
        """
        生成所有候选切片平面的参数
        
        Args:
            center_point: 初始平面中心点
                         - 如果 use_voxel_coord=False: 物理坐标 (x, y, z)，单位：mm
                         - 如果 use_voxel_coord=True: 体素坐标 (i, j, k)
            normal_vector: 初始平面法向量
            translation_range: 法向平移范围（±mm）
            translation_step: 法向平移步长（mm）
            rotation_range: 旋转角度范围（±度）
            rotation_step: 旋转角度步长（度）
            slice_size: 输出切片大小（像素）
            slice_spacing: 切片内像素间距（mm）
            use_voxel_coord: 是否使用体素坐标（默认False，使用物理坐标）
            
        Returns:
            切片参数列表，每个元素包含平面的变换信息
        """
        # 如果使用体素坐标，转换为物理坐标
        if use_voxel_coord:
            center = np.array(self.voxel_to_physical(center_point))
            print(f"体素坐标 {center_point} 转换为物理坐标 {tuple(center)}")
        else:
            center = np.array(center_point)
        
        normal = np.array(normal_vector)
        normal = normal / np.linalg.norm(normal)
        # image_x_axis = direction[:, 0]  # 图像X轴
        image_y_axis = self.direction[:, 1]  # 图像Y轴
        # 构建初始平面的正交坐标系
        # 法向为Z轴，构建X轴和Y轴
        if abs(normal[2]) < 0.9:
            x_axis = np.cross(normal, np.array([0, 0, 1]))
        else:
            x_axis = np.cross(normal, image_y_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # 生成参数网格
        translations = np.arange(-translation_range, translation_range + translation_step/2, translation_step)
        rotations_x = np.arange(-rotation_range, rotation_range + rotation_step/2, rotation_step)
        rotations_y = np.arange(-rotation_range, rotation_range + rotation_step/2, rotation_step)
        # rotations_x = [0]
        # rotations_y = [0]
        slice_planes = []
        plane_id = 0
        
        for trans in translations:
            for rot_x in rotations_x:
                for rot_y in rotations_y:
                    # if plane_id == 122:
                    #     print(f"trans: {trans}, rot_x: {rot_x}, rot_y: {rot_y}")
                    # 计算新的中心点（沿法向平移）
                    new_center = center + trans * normal
                    
                    # 计算旋转后的坐标系
                    # 先绕X轴旋转
                    R_x = self._create_rotation_matrix(x_axis, rot_x)
                    # 再绕Y轴旋转
                    R_y = self._create_rotation_matrix(y_axis, rot_y)
                    # 组合旋转
                    R_combined = np.dot(R_y, R_x)
                    
                    # 应用旋转到坐标系
                    new_x_axis = np.dot(R_combined, x_axis)
                    new_y_axis = np.dot(R_combined, y_axis)
                    new_normal = np.dot(R_combined, normal)
                    
                    slice_planes.append({
                        'id': plane_id,
                        'center': new_center,
                        'x_axis': new_x_axis,
                        'y_axis': new_y_axis,
                        'normal': new_normal,
                        'translation': trans,
                        'rotation_x': rot_x,
                        'rotation_y': rot_y,
                        'size': slice_size,
                        'spacing': slice_spacing
                    })
                    
                    plane_id += 1
        
        print(f"生成了 {len(slice_planes)} 个候选平面")
        return slice_planes
    
    def extract_slice(self, plane_params: dict, label_tag: bool = False) -> np.ndarray:
        """
        从3D图像中提取指定平面的2D切片
        
        Args:
            plane_params: 平面参数字典
            
        Returns:
            2D numpy数组（切片图像）
        """
        center = plane_params['center']
        x_axis = plane_params['x_axis']
        y_axis = plane_params['y_axis']
        normal = plane_params['normal']
        size = plane_params['size']
        spacing = plane_params['spacing']
        
        # 创建重采样器
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([size[0], size[1], 1])
        resampler.SetOutputSpacing([spacing[0], spacing[1], 1.0])
        
        # 构建方向矩阵（列向量为坐标轴）
        direction_matrix = np.column_stack([x_axis, y_axis, normal])
        resampler.SetOutputDirection(direction_matrix.flatten().tolist())
        
        # 设置原点（切片左下角）
        origin = center - (size[0] * spacing[0] / 2.0) * x_axis - (size[1] * spacing[1] / 2.0) * y_axis
        resampler.SetOutputOrigin(origin.tolist())
        

        # label
        if label_tag:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)  
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(-3024)  # CT的空气HU值

        
        # 执行重采样
        slice_image = resampler.Execute(self.image)
        
        # 转换为numpy数组
        slice_array = sitk.GetArrayFromImage(slice_image)
        
        return slice_array.squeeze()
    
    def extract_all_slices(
        self,
        slice_planes: List[dict],
        output_dir: str = None,
        label_tag: bool = False
    ) -> List[np.ndarray]:
        """
        提取所有候选平面的切片
        
        Args:
            slice_planes: 平面参数列表
            output_dir: 输出目录（可选，用于保存切片图像）
            
        Returns:
            切片图像列表
        """
        slices = []
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        csv_data = []
        for i, plane in enumerate(slice_planes):
            # if i != 122:
            #     continue
            print(f"提取切片 {i+1}/{len(slice_planes)}: "
                  f"平移={plane['translation']:.1f}mm, "
                  f"旋转X={plane['rotation_x']:.0f}°, "
                  f"旋转Y={plane['rotation_y']:.0f}°")
            
            slice_array = self.extract_slice(plane, label_tag=label_tag)
            slices.append(slice_array)
            # 保存切片
            if output_dir:
                # 修改文件名扩展名
                filename = f"slice_{i:03d}_t{plane['translation']:.1f}_rx{plane['rotation_x']:.0f}_ry{plane['rotation_y']:.0f}.nii.gz"
                translation = plane['center'] - self.ct_center
                self._save_slice_image(slice_array, output_path / filename, plane_params=plane)
                csv_data.append({'filename': filename, 'translation': translation.tolist()})
        if output_dir:
            import csv
            csv_path = output_path / 'slice_parameters.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'translation']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        return slices
    
    def _save_slice_image(self, slice_array: np.ndarray, filepath: Path, plane_params: dict = None):
        """
        保存切片图像为NII.GZ文件（保留原始HU值和空间信息）
        
        Args:
            slice_array: 切片数组（原始HU值）
            filepath: 保存路径
            plane_params: 平面参数（可选，用于设置正确的spacing和direction）
        """
        # 修改文件扩展名为 .nii.gz
        filepath = filepath.with_suffix('.nii.gz')
        
        # 转换为SimpleITK图像对象
        # 注意：SimpleITK的数组维度顺序是 [z, y, x]，对于2D图像是 [y, x]
        slice_image = sitk.GetImageFromArray(slice_array)
        
        # 如果提供了平面参数，设置正确的空间信息
        if plane_params:
            spacing = plane_params.get('spacing', (1.0, 1.0))
            # 设置2D图像的spacing (x, y)
            slice_image.SetSpacing([spacing[0], spacing[1]])
            
            # 设置origin（平面的左下角）
            center = plane_params['center']
            x_axis = plane_params['x_axis']
            y_axis = plane_params['y_axis']
            size = plane_params['size']
            
            origin = center - (size[0] * spacing[0] / 2.0) * x_axis - (size[1] * spacing[1] / 2.0) * y_axis
            slice_image.SetOrigin(origin.tolist()[:2])
            # slice_image.SetSpacing(spacing)
            
            # 设置direction（2D图像的方向矩阵是2x2）
            direction_2d = np.column_stack([x_axis[:2], y_axis[:2]]).flatten().tolist()
            slice_image.SetDirection(direction_2d)
        
        # 保存为NII.GZ格式，保留原始HU值
        sitk.WriteImage(slice_image, str(filepath), useCompression=True)

    
    def visualize_sample_slices(
        self,
        slices: List[np.ndarray],
        slice_planes: List[dict],
        num_samples: int = 9
    ):
        """
        可视化部分切片样本
        
        Args:
            slices: 切片列表
            slice_planes: 平面参数列表
            num_samples: 显示的样本数量
        """
        indices = np.linspace(0, len(slices)-1, num_samples, dtype=int)
        
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            ax.imshow(slices[idx], cmap='gray', vmin=-1000, vmax=3000)
            
            plane = slice_planes[idx]
            ax.set_title(
                f"ID:{plane['id']}\n"
                f"T:{plane['translation']:.1f}mm\n"
                f"RX:{plane['rotation_x']:.0f}° RY:{plane['rotation_y']:.0f}°",
                fontsize=8
            )
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_slices.png', dpi=150, bbox_inches='tight')
        plt.show()


def generate_for_one_patient(volume_path, output_dir, label_tag, center_point_voxel, normal_vector= (0.0, 0.0, 1.0), use_voxel_coord=True):
    
    # 切片生成参数
    translation_range = 5.0    # ±5mm
    translation_step = 2.5     # 2.5mm步长
    rotation_range = 50      # ±90度
    rotation_step = 25       # 45度步长
    
    # 输出切片的大小和分辨率
    slice_size = (512, 512)           # 像素
    slice_spacing = (0.5, 0.5)        # mm
    
    # ===== 执行切片生成 =====
    
    print("=" * 60)
    print("3D CT切片生成器")
    print("=" * 60)
    
    # 创建生成器
    generator = CT3DSliceGenerator(volume_path)
    
    print(f"\nCT图像信息:")
    print(f"  尺寸: {generator.size} (体素)")
    print(f"  间距: {generator.spacing} mm")
    print(f"  原点: {generator.origin} mm")
    
    # 如果未指定中心点，使用图像中心
    if center_point_voxel is None:
        center_point_voxel = (generator.size[0] / 2, generator.size[1] / 2, generator.size[2] / 2)
        print(f"\n使用图像中心作为初始平面中心:")
    else:
        print(f"\n使用指定的初始平面中心:")
    
    print(f"  体素坐标: {center_point_voxel}")
    if use_voxel_coord:
        center_point_physical = generator.voxel_to_physical(center_point_voxel)
        print(f"  物理坐标: {center_point_physical} mm")
        center_point = center_point_voxel
    else:
        center_point = center_point_voxel
    
    # 生成切片平面参数
    print(f"\n生成切片平面参数...")
    slice_planes = generator.generate_slice_planes(
        center_point=center_point,
        normal_vector=normal_vector,
        translation_range=translation_range,
        translation_step=translation_step,
        rotation_range=rotation_range,
        rotation_step=rotation_step,
        slice_size=slice_size,
        slice_spacing=slice_spacing,
        use_voxel_coord=use_voxel_coord
    )
    
    # 提取所有切片
    print(f"\n开始提取切片...")
    slices = generator.extract_all_slices(
        slice_planes=slice_planes,
        output_dir=output_dir,
        label_tag=label_tag
    )
    
    print(f"\n切片提取完成！共生成 {len(slices)} 个切片")
    print(f"切片已保存到: {output_dir}/")
    
    # # 可视化部分样本
    # print(f"\n生成可视化样本...")
    # generator.visualize_sample_slices(slices, slice_planes, num_samples=9)
    # print(f"样本可视化已保存: sample_slices.png")
    
    # print("\n" + "=" * 60)
    # print("完成！")
    # print("=" * 60)


if __name__ == "__main__":
    import os
    from LV_apex_cal import find_annulus, calculate_mitral_annulus_center

    image_path = r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\CT_resampled_nii"
    label_path = r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\CT_segmentation"
    output_path = r"D:\dataset\TEECT_data\ct_paired"
    for file in os.listdir(image_path):
        patient_id = file.split(".")[0]
        if patient_id != "Patient_0006":
            continue
        output_image_dir = os.path.join(output_path, patient_id + "_image")
        output_label_dir = os.path.join(output_path, patient_id + "_label")

        if os.path.exists(output_image_dir) and os.path.exists(output_label_dir):
            continue
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        label_array = sitk.ReadImage(os.path.join(label_path, patient_id, f"{patient_id}_label.nii.gz"))
        label_array = sitk.GetArrayFromImage(label_array)
        label_dict = {"rv_label": 1, "ra_label": 2, "lv_label": 4, "la_label": 3}
        mitral_points = find_annulus(label_array, v_label=label_dict["lv_label"], a_label=label_dict["la_label"])
        mitral_center = calculate_mitral_annulus_center(mitral_points)
        mitral_center = tuple(mitral_center.astype(float))[::-1]
        generate_for_one_patient(
            volume_path=os.path.join(image_path, file),
            output_dir=output_image_dir,
            label_tag=False,
            center_point_voxel=mitral_center
        )
        generate_for_one_patient(
            volume_path=os.path.join(label_path, patient_id, f"{patient_id}_label.nii.gz"),
            output_dir=output_label_dir,
            label_tag=True,
            center_point_voxel=mitral_center
        )