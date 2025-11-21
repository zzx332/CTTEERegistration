"""
2D超声到3D CT的基于互信息的配准
使用8参数变换：3个旋转角 + 3个平移 + 2个缩放
"""

import SimpleITK as sitk
import numpy as np
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable
import time


class TwoD_ThreeD_Registration:
    """
    2D-3D配准类：2D超声图像配准到3D CT体数据
    """
    
    def __init__(self):
        """初始化"""
        self.ct_volume = None
        self.ultrasound_2d = None
        self.ct_mask = None  # 新增：CT心脏mask
        self.best_params = None
        self.optimization_history = []
        
    def load_images(self, ct_path: str, ultrasound_path: str, ct_mask_path: Optional[str] = None, us_mask_path: Optional[str] = None):
        """
        加载3D CT和2D超声图像
        
        Args:
            ct_path: 3D CT体数据路径
            ultrasound_path: 2D超声图像路径
            ct_mask_path: CT心脏mask路径（可选）
        """
        print("加载图像...")
        self.ct_volume = sitk.ReadImage(ct_path)
        self.ultrasound_2d = sitk.ReadImage(ultrasound_path)
        
        # 加载mask（如果提供）
        if ct_mask_path is not None:
            print(f"加载CT mask: {ct_mask_path}")
            self.ct_mask = sitk.ReadImage(ct_mask_path)
            print(f"  Mask尺寸: {self.ct_mask.GetSize()}")
            
            # 验证mask和CT尺寸一致
            if self.ct_mask.GetSize() != self.ct_volume.GetSize():
                raise ValueError(f"Mask尺寸 {self.ct_mask.GetSize()} 与 CT尺寸 {self.ct_volume.GetSize()} 不匹配！")
        else:
            self.ct_mask = None
            print("  未提供CT mask，将使用全图计算互信息")
        
        if us_mask_path is not None:
            print(f"加载超声 mask: {us_mask_path}")
            self.us_mask = sitk.ReadImage(us_mask_path)
            print(f"  Mask尺寸: {self.us_mask.GetSize()}")
            
            # 验证mask和超声尺寸一致
            if self.us_mask.GetSize() != self.ultrasound_2d.GetSize():
                raise ValueError(f"Mask尺寸 {self.us_mask.GetSize()} 与超声尺寸 {self.ultrasound_2d.GetSize()} 不匹配！")
        else:
            self.us_mask = None
            print("  未提供超声 mask，将使用全图计算互信息")
        
        print(f"  CT尺寸: {self.ct_volume.GetSize()}")
        print(f"  CT spacing: {self.ct_volume.GetSpacing()}")
        print(f"  超声尺寸: {self.ultrasound_2d.GetSize()}")
        print(f"  超声spacing: {self.ultrasound_2d.GetSpacing()}")
        
    def extract_slice_from_volume(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        tx: float,
        ty: float,
        tz: float,
        sx: float = 1.0,
        sy: float = 1.0,
        slice_size: Optional[Tuple[int, int]] = None,
        slice_spacing: Optional[Tuple[float, float]] = None,
        extract_mask: bool = False  # 新增参数：是否同时提取mask
    ) -> sitk.Image:
        """
        根据变换参数从3D CT中提取2D切片
        """
        # 默认使用超声图像的尺寸和spacing
        if slice_size is None:
            slice_size = self.ultrasound_2d.GetSize()
        if slice_spacing is None:
            slice_spacing = self.ultrasound_2d.GetSpacing()
        
        # ===== 修复1：确保spacing始终为正 =====
        slice_spacing_scaled = [
            abs(slice_spacing[0] / sx),  # 取绝对值
            abs(slice_spacing[1] / sy),  # 取绝对值
            1.0
        ]
        
        # 构建旋转矩阵（3x3）
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)
        
        Rx = np.array([
            [1,      0,       0     ],
            [0,      cos_a,  -sin_a],
            [0,      sin_a,   cos_a]
        ])

        Ry = np.array([
            [cos_b,  0,       sin_b],
            [0,      1,       0    ],
            [-sin_b, 0,       cos_b]
        ])

        Rz = np.array([
            [cos_g, -sin_g,  0],
            [sin_g,  cos_g,  0],
            [0,      0,      1]
        ])

        # 组合旋转
        R = Rz @ Ry @ Rx
        
        # ===== 修复2：验证旋转矩阵的正交性和行列式 =====
        det_R = np.linalg.det(R)
        orthogonality_error = np.max(np.abs(R @ R.T - np.eye(3)))
        
        # 调试信息（仅在第一次调用时打印）
        if not hasattr(self, '_debug_printed'):
            print(f"  旋转矩阵行列式: {det_R:.6f} (应该≈1)")
            print(f"  正交性误差: {orthogonality_error:.2e} (应该≈0)")
            self._debug_printed = True
        
        # 如果行列式为负，说明有镜像，需要修正
        if det_R < 0:
            print(f"  ⚠️ 警告：旋转矩阵行列式为负 ({det_R:.6f})，进行修正...")
            # 翻转Z轴方向
            R[:, 2] = -R[:, 2]
            det_R = np.linalg.det(R)
            print(f"  修正后行列式: {det_R:.6f}")
        
        # 如果正交性误差太大，重新正交化
        if orthogonality_error > 1e-6:
            print(f"  ⚠️ 警告：旋转矩阵不正交，进行Gram-Schmidt正交化...")
            # Gram-Schmidt正交化
            x_axis = R[:, 0]
            y_axis = R[:, 1]
            z_axis = R[:, 2]
            
            # 正交化
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)  # 确保右手系
            
            R = np.column_stack([x_axis, y_axis, z_axis])
            print(f"  正交化后行列式: {np.linalg.det(R):.6f}")

        # CT中心点
        ct_size = np.array(self.ct_volume.GetSize())
        ct_spacing = np.array(self.ct_volume.GetSpacing())
        ct_origin = np.array(self.ct_volume.GetOrigin())
        ct_center = ct_origin + ct_size * ct_spacing / 2.0
        
        # 切片中心（在3D空间中，经过旋转和平移后）
        slice_center = ct_center + np.array([tx, ty, tz])
        
        # 切片的方向向量（旋转后的坐标轴）
        x_axis = R[:, 0]  # 切片的X轴方向（列方向）
        y_axis = R[:, 1]  # 切片的Y轴方向（行方向）
        z_axis = R[:, 2]  # 切片的法向量（Z轴方向）
        
        # 计算切片的origin（左上角或左下角的物理坐标）
        half_width = (slice_size[0] - 1) * slice_spacing_scaled[0] / 2.0
        half_height = (slice_size[1] - 1) * slice_spacing_scaled[1] / 2.0
        
        slice_origin = slice_center - half_width * x_axis - half_height * y_axis
        
        # 构建方向矩阵（列向量为坐标轴）
        direction_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # ===== 修复3：最终验证 =====
        final_det = np.linalg.det(direction_matrix)
        if final_det < 0:
            raise ValueError(f"方向矩阵行列式为负 ({final_det:.6f})，无法创建有效的图像空间！")
        
        # 设置Resample参数
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([slice_size[0], slice_size[1], 1])
        resampler.SetOutputSpacing(slice_spacing_scaled)
        resampler.SetOutputOrigin(slice_origin.tolist())
        resampler.SetOutputDirection(direction_matrix.flatten().tolist())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-3024)
        resampler.SetTransform(sitk.Transform())
        # 提取3D切片（尺寸为 W×H×1）
        extracted_3d = resampler.Execute(self.ct_volume)
        
        # 降维到2D（尺寸为 W×H）
        extractor = sitk.ExtractImageFilter()
        size = list(extracted_3d.GetSize())
        size[2] = 0
        extractor.SetSize(size)
        extractor.SetIndex([0, 0, 0])
        
        extracted_2d = extractor.Execute(extracted_3d)
        sitk.WriteImage(extracted_2d, r"D:\dataset\TEECT_data\ct\temp.nii.gz", useCompression=True)
        extracted_2d = sitk.ReadImage(r"D:\dataset\TEECT_data\ct\temp.nii.gz")
        # 如果需要提取mask且mask存在
        if extract_mask and self.ct_mask is not None:
            # 使用相同的resampler参数提取mask
            resampler_mask = sitk.ResampleImageFilter()
            resampler_mask.SetSize([slice_size[0], slice_size[1], 1])
            resampler_mask.SetOutputSpacing(slice_spacing_scaled)
            resampler_mask.SetOutputOrigin(slice_origin.tolist())
            resampler_mask.SetOutputDirection(direction_matrix.flatten().tolist())
            resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)  # mask用最近邻插值
            resampler_mask.SetDefaultPixelValue(0)  # 背景为0
            resampler_mask.SetTransform(sitk.Transform())
            
            # 提取mask的3D切片
            mask_3d = resampler_mask.Execute(self.ct_mask)
            
            # 降维到2D
            mask_2d = extractor.Execute(mask_3d)
            sitk.WriteImage(mask_2d, r"D:\dataset\TEECT_data\ct\temp.nii.gz", useCompression=True)
            mask_2d = sitk.ReadImage(r"D:\dataset\TEECT_data\ct\temp.nii.gz")
            return extracted_2d, mask_2d
        
        return extracted_2d

            
    def compute_chamfer_distance(
        self,
        edge1: np.ndarray,
        edge2: np.ndarray,
        bidirectional: bool = True
    ) -> float:
        """
        计算两个边缘点云之间的Chamfer距离
        
        Args:
            edge1: 边缘点云1 (N1, 2)，物理坐标
            edge2: 边缘点云2 (N2, 2)，物理坐标
            bidirectional: 是否计算双向距离
        
        Returns:
            chamfer_dist: Chamfer距离（越小越好）
        """
        from scipy.spatial.distance import cdist
        
        if len(edge1) == 0 or len(edge2) == 0:
            return 1e6  # 如果没有边缘点，返回大惩罚值
        
        # 计算距离矩阵
        dist_matrix = cdist(edge1, edge2, metric='euclidean')
        
        # edge1到edge2的最近点距离
        dist_1to2 = np.min(dist_matrix, axis=1)  # (N1,)
        mean_dist_1to2 = np.mean(dist_1to2)
        
        if bidirectional:
            # edge2到edge1的最近点距离
            dist_2to1 = np.min(dist_matrix, axis=0)  # (N2,)
            mean_dist_2to1 = np.mean(dist_2to1)
            
            # 双向Chamfer距离（对称）
            chamfer_dist = (mean_dist_1to2 + mean_dist_2to1) / 2.0
        else:
            chamfer_dist = mean_dist_1to2
        
        return chamfer_dist

    def compute_centroid_distance(
        self, 
        edge1: np.ndarray, 
        edge2: np.ndarray
    ) -> float:
        """
        计算两个点云质心之间的距离。
        这是一个非常简单的全局距离度量，只考虑平移。
        """
        if len(edge1) == 0 or len(edge2) == 0:
            return 1e6

        centroid1 = np.mean(edge1, axis=0)
        centroid2 = np.mean(edge2, axis=0)
        
        distance = np.linalg.norm(centroid1 - centroid2)
        return distance

    def compute_principal_axis_distance(
        self, 
        edge1: np.ndarray, 
        edge2: np.ndarray,
        weight_translation: float = 0.5,
        weight_rotation: float = 0.5
    ) -> float:
        """
        通过主轴分析（PCA）计算两个点云的“整体”距离。
        该方法对整体形状的平移和旋转都敏感。

        返回一个结合了质心距离和主方向夹角的加权代价值。
        """
        if len(edge1) < 2 or len(edge2) < 2:
            return 1e6

        # 1. 计算质心和质心距离
        centroid1 = np.mean(edge1, axis=0)
        centroid2 = np.mean(edge2, axis=0)
        dist_centroids = np.linalg.norm(centroid1 - centroid2)

        # 2. 中心化点云
        centered_edge1 = edge1 - centroid1
        centered_edge2 = edge2 - centroid2

        # 3. 计算协方差矩阵
        cov1 = np.cov(centered_edge1, rowvar=False)
        cov2 = np.cov(centered_edge2, rowvar=False)

        # 4. 计算特征向量（即主轴）
        # eigh返回的特征向量是归一化的，并按特征值大小排序
        _, eigvecs1 = np.linalg.eigh(cov1)
        _, eigvecs2 = np.linalg.eigh(cov2)

        # 第一个主轴是对应最大特征值的特征向量
        principal_axis1 = eigvecs1[:, -1]
        principal_axis2 = eigvecs2[:, -1]

        # 5. 计算主轴之间的夹角（弧度）
        # 使用点积的绝对值，因为特征向量的方向可以是相反的
        cos_angle = abs(np.dot(principal_axis1, principal_axis2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0) # 避免浮点误差
        angle_rad = np.arccos(cos_angle) # 范围 [0, pi/2]

        # 6. 归一化并加权组合
        # 将距离和角度归一化到相似的范围，以平衡它们的影响
        # 假设图像对角线长度作为典型距离
        img_diag = 200.0 # mm, 一个典型值，可以根据实际情况调整
        normalized_dist = dist_centroids / img_diag
        
        # 角度的最大值是 pi/2
        normalized_angle = angle_rad / (np.pi / 2.0)
        
        # 组合成最终的代价值
        cost = (weight_translation * normalized_dist + 
                weight_rotation * normalized_angle)
                
        return cost


    def extract_edge_points(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None,
        method: str = 'canny',
        subsample: int = 5
    ) -> np.ndarray:
        """
        从图像或mask中提取边缘点的物理坐标
        
        Args:
            image: SimpleITK图像
            mask: 如果提供mask，从mask轮廓提取边缘；否则从图像提取
            method: 边缘检测方法 ('canny', 'sobel', 'mask_contour')
            subsample: 边缘点下采样间隔（每隔N个点取1个）
        
        Returns:
            edge_points: (N, 2) 边缘点的物理坐标
        """
        import cv2
        from skimage import morphology
        
        # 转换为numpy数组
        if mask is not None:
            # 从mask提取轮廓
            arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
            
            # 如果是多标签，二值化（>0为前景）
            if arr.max() > 1:
                arr = (arr > 0).astype(np.uint8)
            
            # 提取轮廓（只要边界）
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(arr, kernel, iterations=1)
            edge_arr = arr - eroded  # 边界 = 原图 - 腐蚀
            
        else:
            # 从图像提取边缘
            arr = sitk.GetArrayFromImage(image)
            
            if method == 'canny':
                # 归一化到0-255
                arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
                edge_arr = cv2.Canny(arr_norm, 50, 150)
                edge_arr = (edge_arr > 0).astype(np.uint8)
                
            elif method == 'sobel':
                from scipy import ndimage
                # Sobel梯度
                grad_x = ndimage.sobel(arr, axis=1)
                grad_y = ndimage.sobel(arr, axis=0)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                # 阈值化
                threshold = np.percentile(grad_mag, 95)
                edge_arr = (grad_mag > threshold).astype(np.uint8)
        
        # 细化边缘（可选，使边缘更细）
        edge_arr = morphology.skeletonize(edge_arr > 0).astype(np.uint8)
        
        # 获取边缘点的像素坐标
        edge_coords = np.argwhere(edge_arr > 0)  # (N, 2) [row, col]
        
        # 下采样
        if subsample > 1 and len(edge_coords) > 0:
            indices = np.arange(0, len(edge_coords), subsample)
            edge_coords = edge_coords[indices]
        
        # 如果边缘点太少，返回空
        if len(edge_coords) < 10:
            return np.array([]).reshape(0, 2)
        
        # 转换为物理坐标
        spacing = np.array(image.GetSpacing())
        origin = np.array(image.GetOrigin())
        direction = np.array(image.GetDirection()).reshape(2, 2)
        
        # 像素坐标 -> 物理坐标
        # physical = origin + direction @ (spacing * pixel_coords)
        edge_physical = []
        for row, col in edge_coords:
            # SimpleITK: (col, row) = (x, y)
            pixel = np.array([col, row])
            physical = origin + direction @ (spacing * pixel)
            edge_physical.append(physical)
        
        return np.array(edge_physical)  # (N, 2)


    def objective_function(self, params: np.ndarray) -> float:
        """
        目标函数：返回边缘点云之间的Chamfer距离（越小越好）
        """
        alpha, beta, gamma, tx, ty, tz, sx, sy = params
        
        try:
            # 提取CT切片和mask
            if self.ct_mask is not None:
                extracted_slice, mask_slice = self.extract_slice_from_volume(
                    alpha, beta, gamma, tx, ty, tz, sx, sy, extract_mask=True
                )
            else:
                extracted_slice = self.extract_slice_from_volume(
                    alpha, beta, gamma, tx, ty, tz, sx, sy, extract_mask=False
                )
                mask_slice = None
            
            # 从CT切片提取边缘点（优先使用mask轮廓）
            if mask_slice is not None:
                ct_edge_points = self.extract_edge_points(
                    extracted_slice, mask=mask_slice, 
                    method='mask_contour', subsample=5
                )
            else:
                ct_edge_points = self.extract_edge_points(
                    extracted_slice, mask=None, 
                    method='canny', subsample=5
                )
            
            # 从超声图像提取边缘点
            # 如果超声也有mask，可以从mask提取；否则用边缘检测
            us_edge_points = self.extract_edge_points(
                self.ultrasound_2d, mask=self.us_mask,  # 如果有超声mask，替换为 mask=self.us_mask
                method='canny', subsample=5
            )
            
            # 检查边缘点数量
            if len(ct_edge_points) < 10 or len(us_edge_points) < 10:
                print(f"    ⚠️ 边缘点过少: CT={len(ct_edge_points)}, US={len(us_edge_points)}")
                return 1e6
            
            # 计算Chamfer距离
            # chamfer_dist = self.compute_chamfer_distance(
            #     ct_edge_points, us_edge_points, bidirectional=True
            # )
            chamfer_dist = self.compute_principal_axis_distance(
            ct_edge_points, us_edge_points,
            weight_translation=0.5,  # 平移权重
            weight_rotation=0.5    # 旋转权重
            )
            # 记录历史
            self.optimization_history.append({
                'params': params.copy(),
                'chamfer_dist': chamfer_dist,
                'n_ct_points': len(ct_edge_points),
                'n_us_points': len(us_edge_points)
            })
            
            # 打印当前最优值
            if len(self.optimization_history) % 10 == 0:
                print(f"  迭代 {len(self.optimization_history)}: Chamfer={chamfer_dist:.3f}mm, "
                    f"角度=({np.degrees(alpha):.1f}°, {np.degrees(beta):.1f}°, {np.degrees(gamma):.1f}°), "
                    f"CT点={len(ct_edge_points)}, US点={len(us_edge_points)},"
                    f"平移=({tx:.1f}, {ty:.1f}, {tz:.1f}),"
                    f"缩放=({sx:.1f}, {sy:.1f})")
            
            # 返回Chamfer距离（最小化）
            return chamfer_dist
            
        except Exception as e:
            print(f"  计算失败: {e}")
            import traceback
            traceback.print_exc()
            return 1e6

    def objective_normalized(self,params_norm):
        """接受归一化参数，内部转换为真实参数"""
        # 反归一化
        params_real = params_norm.copy()
        params_real = params_norm * self.param_scales
        # params_real[6:] = params_norm[6:] * self.param_scales[6:] + 1.0
        
        # 调用原始目标函数
        return self.objective_function(params_real)


    def register(
        self,
        initial_params: Optional[np.ndarray] = None,
        method: str = 'powell',
        bounds: Optional[list] = None,
        max_iterations: int = 200
    ) -> Tuple[np.ndarray, dict]:
        """
        执行配准
        
        Args:
            initial_params: 初始参数 [alpha, beta, gamma, tx, ty, tz, sx, sy]
                          如果为None，则从粗略位置开始
            method: 优化方法 ('powell', 'nelder-mead', 'de' for differential evolution)
            bounds: 参数边界 [(min, max), ...] for each parameter
            max_iterations: 最大迭代次数
        
        Returns:
            best_params: 最优参数
            result_dict: 结果字典
        """
        print(f"\n开始2D-3D配准...")
        print(f"  优化方法: {method}")
        
        # 默认初始参数
        if initial_params is None:
            initial_params = np.array([
                np.radians(0), np.radians(0), np.radians(0),  # 角度
                0.0, 0.0, 0.0,  # 平移
                1.0, 1.0        # 缩放
            ])
        
        # ===== 定义参数尺度 =====
        # 将所有参数归一化到相似的数量级（例如都在[-1, 1]范围内）
        # self.param_scales = np.array([
        #     np.pi * 50/180,   # alpha: ±30° ≈ ±0.52 rad → 归一化到 ±1
        #     np.pi * 50/180,   # beta: ±30°
        #     np.pi,     # gamma: ±180° ≈ ±3.14 rad → 归一化到 ±1
        #     50.0,      # tx: ±50mm → 归一化到 ±1
        #     50.0,      # ty: ±50mm
        #     50.0,      # tz: ±50mm
        # ])
        self.param_scales = np.array([
            1.0,   # alpha: ±30° ≈ ±0.52 rad → 归一化到 ±1
            1.0,   # beta: ±30°
            1.0,     # gamma: ±180° ≈ ±3.14 rad → 归一化到 ±1
            10.0,      # tx: ±50mm → 归一化到 ±1
            10.0,      # ty: ±50mm
            10.0,      # tz: ±50mm
        ])
        # self.param_scales = np.array([
        #     0.1*1.0,   # alpha: ±30° ≈ ±0.52 rad → 归一化到 ±1
        #     0.1*1.0,   # beta: ±30°
        #     0.1*1.0,     # gamma: ±180° ≈ ±3.14 rad → 归一化到 ±1
        #     1.0,      # tx: ±50mm → 归一化到 ±1
        #     1.0,      # ty: ±50mm
        #     1.0,      # tz: ±50mm
        # ])

        # 归一化初始参数
        initial_params_norm = initial_params.copy()
        initial_params_norm = initial_params / self.param_scales
        
        print(f"  原始初始参数: {initial_params}")
        print(f"  归一化初始参数: {initial_params_norm}")
        
        # 归一化边界
        if bounds is None:
            # 默认边界（原始空间）
            bounds_original = [
                (-np.pi * 50/180, np.pi* 50/180),   # alpha
                (-np.pi* 50/180, np.pi* 50/180),   # beta
                (-np.pi, np.pi),       # gamma
                (-50, 50),             # tx
                (-50, 50),             # ty
                (-50, 50),             # tz
                (0.8, 1.2),            # sx
                (0.8, 1.2)             # sy
            ]
        else:
            bounds_original = bounds
        
        # 转换为归一化边界
        bounds_norm = []
        for i, (lb, ub) in enumerate(bounds_original):
            if i < 6:
                # 角度和平移
                lb_norm = lb / self.param_scales[i]
                ub_norm = ub / self.param_scales[i]
            else:
                # 缩放
                lb_norm = lb
                ub_norm = ub
            bounds_norm.append((lb_norm, ub_norm))
        start_time = time.time()
        
        # 根据方法选择优化器
        if method.lower() == 'de':
            # 差分进化（全局优化）
            print("  使用差分进化算法（全局搜索）...")
            result = differential_evolution(
                self.objective_normalized,
                bounds=bounds_norm,
                maxiter=max_iterations,
                popsize=15,
                atol=0.001,
                tol=0.001,
                workers=1
            )
            best_params = result.x
            final_mi = -result.fun
            
        else:
            # 局部优化
            print(f"  使用{method}算法（局部优化）...")
            result = minimize(
                self.objective_normalized,
                x0=initial_params_norm,
                method=method,
                bounds=bounds_norm if method in ['l-bfgs-b', 'tnc', 'slsqp'] else None,
                options={'maxiter': max_iterations,
                'maxfev':max_iterations*1}
            )
            # 反归一化最优参数
            best_params_norm = result.x
            best_params = best_params_norm.copy()
            best_params = best_params_norm * self.param_scales
            # best_params[6:] = best_params_norm[6:] * self.param_scales[6:] + 1.0
            final_mi = -result.fun
        
        elapsed_time = time.time() - start_time
        
        # 保存最优参数
        self.best_params = best_params
        
        # 提取最优切片
        best_slice = self.extract_slice_from_volume(*best_params)
        
        # 准备结果
        result_dict = {
            'best_params': best_params,
            'alpha_deg': np.degrees(best_params[0]),
            'beta_deg': np.degrees(best_params[1]),
            'gamma_deg': np.degrees(best_params[2]),
            'translation': best_params[3:6],
            'scaling': best_params[6:8],
            'final_mi': final_mi,
            'num_iterations': len(self.optimization_history),
            'elapsed_time': elapsed_time,
            'best_slice': best_slice
        }
        
        print(f"\n配准完成！")
        print(f"  迭代次数: {result_dict['num_iterations']}")
        print(f"  最终MI: {final_mi:.4f}")
        print(f"  旋转角度: α={result_dict['alpha_deg']:.2f}°, "
              f"β={result_dict['beta_deg']:.2f}°, γ={result_dict['gamma_deg']:.2f}°")
        print(f"  平移: ({best_params[3]:.2f}, {best_params[4]:.2f}, {best_params[5]:.2f}) mm")
        print(f"  缩放: ({best_params[6]:.3f}, {best_params[7]:.3f})")
        print(f"  耗时: {elapsed_time:.1f}秒")
        
        return best_params, result_dict
    
    def register_icp(
        self,
        initial_params: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        max_correspondence_dist: float = 50.0,
        inner_opt_iterations: int = 10
    ) -> Tuple[np.ndarray, dict]:
        """
        使用ICP风格的迭代优化进行2D-3D配准
        
        Args:
            initial_params: 初始参数 [alpha, beta, gamma, tx, ty, tz, sx, sy]
            max_iterations: 最大ICP外层迭代次数
            tolerance: 收敛阈值（平均距离变化）
            max_correspondence_dist: 最大对应点距离阈值（mm）
            inner_opt_iterations: 每次ICP迭代内部的优化迭代次数
        
        Returns:
            best_params: 最优参数
            result_dict: 结果字典
        """
        print(f"\n开始2D-3D ICP配准...")
        print(f"  最大迭代次数: {max_iterations}")
        print(f"  最大对应距离: {max_correspondence_dist}mm")
        
        from scipy.spatial import cKDTree
        
        start_time = time.time()
        
        # 默认初始参数
        if initial_params is None:
            initial_params = np.array([
                np.radians(0), np.radians(0), np.radians(0),  # 角度
                0.0, 0.0, 0.0,  # 平移
                1.0, 1.0        # 缩放
            ])
        self.param_scales = np.array([
            1.0,   # alpha: ±30° ≈ ±0.52 rad → 归一化到 ±1
            1.0,   # beta: ±30°
            1.0,     # gamma: ±180° ≈ ±3.14 rad → 归一化到 ±1
            10.0,      # tx: ±50mm → 归一化到 ±1
            10.0,      # ty: ±50mm
            10.0,      # tz: ±50mm
            1.0,      # sx: ±1
            1.0,      # sy: ±1
        ])
        initial_params_norm = initial_params.copy()
        initial_params_norm = initial_params / self.param_scales

        # params = initial_params.copy()
        params = initial_params_norm.copy()
        self.optimization_history = []
        
        # 提取超声边缘点（固定，只需提取一次）
        print("  提取超声边缘点...")
        us_edge_points = self.extract_edge_points(
            self.ultrasound_2d, 
            mask=self.us_mask if hasattr(self, 'us_mask') else None,
            method='canny', 
            subsample=3  # 更密集的采样
        )
        
        if len(us_edge_points) < 10:
            raise ValueError(f"超声边缘点过少: {len(us_edge_points)}")
        
        print(f"  超声边缘点数: {len(us_edge_points)}")
        
        prev_mean_dist = float('inf')
        best_params = params.copy()
        best_cost = float('inf')
        
        # 保存初始边缘点（用于可视化）
        initial_ct_edge_points = None
        initial_params_saved = initial_params.copy()
        # ICP主循环
        for icp_iter in range(max_iterations):
            print(f"\n--- ICP迭代 {icp_iter + 1}/{max_iterations} ---")
            
            # 步骤1: 提取当前参数对应的CT切片边缘点 
            # alpha, beta, gamma, tx, ty, tz, sx, sy = params
            alpha, beta, gamma, tx, ty, tz, sx, sy = params * self.param_scales
            
            try:
                if self.ct_mask is not None:
                    extracted_slice, mask_slice = self.extract_slice_from_volume(
                        alpha, beta, gamma, tx, ty, tz, sx, sy, extract_mask=True
                    )
                    ct_edge_points = self.extract_edge_points(
                        extracted_slice, mask=mask_slice, 
                        method='mask_contour', subsample=3
                    )
                else:
                    extracted_slice = self.extract_slice_from_volume(
                        alpha, beta, gamma, tx, ty, tz, sx, sy, extract_mask=False
                    )
                    ct_edge_points = self.extract_edge_points(
                        extracted_slice, mask=None, 
                        method='canny', subsample=3
                    )
            except Exception as e:
                print(f"  ⚠️ 提取切片失败: {e}")
                break
            
            if len(ct_edge_points) < 10:
                print(f"  ⚠️ CT边缘点过少: {len(ct_edge_points)}")
                break
            
            # 保存第一次迭代的CT边缘点（初始状态）
            if icp_iter == 0:
                initial_ct_edge_points = ct_edge_points.copy()
            
            # 步骤2: 建立点对应关系（CT点 -> US点的最近邻）
            us_tree = cKDTree(us_edge_points)
            distances, indices = us_tree.query(ct_edge_points)
            
            # 过滤距离过大的对应点（outlier rejection）
            valid_mask = distances < max_correspondence_dist
            num_valid = np.sum(valid_mask)
            
            if num_valid < 10:
                print(f"  ⚠️ 有效对应点过少: {num_valid}")
                break
            
            ct_points_valid = ct_edge_points[valid_mask]
            us_points_valid = us_edge_points[indices[valid_mask]]
            valid_distances = distances[valid_mask]
            mean_dist = np.mean(valid_distances)
            
            print(f"  CT边缘点数: {len(ct_edge_points)}")
            print(f"  有效对应点对: {num_valid}")
            print(f"  平均对应距离: {mean_dist:.3f}mm")
            print(f"  当前参数: α={np.degrees(alpha):.1f}°, β={np.degrees(beta):.1f}°, "
                  f"γ={np.degrees(gamma):.1f}°, t=({tx:.1f},{ty:.1f},{tz:.1f})mm")
            
            # 步骤3: 基于固定的点对应关系优化参数
            # 定义局部代价函数：给定固定的目标点us_points_valid，优化参数
            def correspondence_cost(p):
                """给定参数，提取CT边缘，计算与固定US目标点的距离"""
                # alpha_, beta_, gamma_, tx_, ty_, tz_, sx_, sy_ = p
                alpha_, beta_, gamma_, tx_, ty_, tz_, sx_, sy_ = p * self.param_scales
                try:
                    if self.ct_mask is not None:
                        slice_, mask_ = self.extract_slice_from_volume(
                            alpha_, beta_, gamma_, tx_, ty_, tz_, sx_, sy_, extract_mask=True
                        )
                        ct_pts = self.extract_edge_points(
                            slice_, mask=mask_, method='mask_contour', subsample=3
                        )
                    else:
                        slice_ = self.extract_slice_from_volume(
                            alpha_, beta_, gamma_, tx_, ty_, tz_, sx_, sy_, extract_mask=False
                        )
                        ct_pts = self.extract_edge_points(
                            slice_, mask=None, method='canny', subsample=3
                        )
                    
                    if len(ct_pts) < 10:
                        return 1e6
                    
                    # 计算CT点到固定US点集的最近距离
                    tree = cKDTree(us_points_valid)
                    dists, _ = tree.query(ct_pts)
                    
                    # 只考虑距离小于阈值的点
                    valid_dists = dists[dists < max_correspondence_dist]
                    if len(valid_dists) < 5:
                        return 1e6
                    
                    return np.mean(valid_dists)
                    
                except Exception as e:
                    return 1e6
            
            # 使用局部优化器优化参数
            print(f"  优化参数（内部迭代{inner_opt_iterations}次）...")
            result = minimize(
                correspondence_cost,
                x0=params,
                method='Powell',
                options={
                    'maxiter': inner_opt_iterations,
                    'maxfev': inner_opt_iterations * 2
                }
            )
            
            # 更新参数
            params = result.x
            # params = params * self.param_scales
            current_cost = result.fun
            
            print(f"  优化后代价: {current_cost:.3f}mm")
            
            # 记录历史
            self.optimization_history.append({
                'icp_iteration': icp_iter + 1,
                'params': params.copy(),
                'mean_distance': mean_dist,
                'num_correspondences': num_valid,
                'optimized_cost': current_cost
            })
            
            # 更新最优参数
            if current_cost < best_cost:
                best_cost = current_cost
                best_params = params.copy()
                print(f"  ✓ 更新最优参数")
            
            # 步骤4: 检查收敛
            dist_change = abs(prev_mean_dist - mean_dist)
            # if dist_change < tolerance:
            #     print(f"\n✓ 收敛！距离变化 {dist_change:.4f}mm < {tolerance}mm")
            #     break
            # 在函数参数中添加
            min_iterations: int = 20 

            # 在收敛检查中：
            if dist_change < tolerance and icp_iter >= min_iterations:
                print(f"\n✓ 收敛！距离变化 {dist_change:.4f}mm < {tolerance}mm (迭代{icp_iter+1}次)")
                break
            elif dist_change < tolerance:
                print(f"  距离变化小于阈值，但未达到最小迭代次数 ({icp_iter+1}/{min_iterations})")
        
        elapsed_time = time.time() - start_time
        
        # 保存最优参数
        self.best_params = best_params
        
        # 提取最优切片和最终边缘点
        # best_slice = self.extract_slice_from_volume(*best_params)
        best_slice = self.extract_slice_from_volume(*best_params * self.param_scales)
        
        # 提取最终的CT边缘点
        try:
            if self.ct_mask is not None:
                final_slice, final_mask = self.extract_slice_from_volume(
                    *best_params * self.param_scales, extract_mask=True
                )
                final_ct_edge_points = self.extract_edge_points(
                    final_slice, mask=final_mask, 
                    method='mask_contour', subsample=3
                )
            else:
                final_slice = self.extract_slice_from_volume(*best_params * self.param_scales, extract_mask=False)
                final_ct_edge_points = self.extract_edge_points(
                    final_slice, mask=None, 
                    method='canny', subsample=3
                )
        except:
            final_ct_edge_points = None
        
        # 准备结果
        result_dict = {
            'best_params': best_params,
            'initial_params': initial_params_saved,
            'alpha_deg': np.degrees(best_params[0] * self.param_scales[0]),
            'beta_deg': np.degrees(best_params[1] * self.param_scales[1]),
            'gamma_deg': np.degrees(best_params[2] * self.param_scales[2]),
            'translation': best_params[3:6] * self.param_scales[3:6],
            'scaling': best_params[6:8] * self.param_scales[6:8],
            'final_cost': best_cost,
            'num_iterations': len(self.optimization_history),
            'elapsed_time': elapsed_time,
            'best_slice': best_slice,
            'us_edge_points': us_edge_points,
            'initial_ct_edge_points': initial_ct_edge_points,
            'final_ct_edge_points': final_ct_edge_points
        }
        
        print(f"\n{'='*60}")
        print(f"ICP配准完成！")
        print(f"  总迭代次数: {result_dict['num_iterations']}")
        print(f"  最终代价: {best_cost:.3f}mm")
        print(f"  旋转角度: α={result_dict['alpha_deg']:.2f}°, "
              f"β={result_dict['beta_deg']:.2f}°, γ={result_dict['gamma_deg']:.2f}°")
        print(f"  平移: ({best_params[3] * self.param_scales[3]:.2f}, {best_params[4] * self.param_scales[4]:.2f}, {best_params[5] * self.param_scales[5]:.2f}) mm")
        print(f"  缩放: ({best_params[6] * self.param_scales[6]:.3f}, {best_params[7] * self.param_scales[7]:.3f})")
        print(f"  耗时: {elapsed_time:.1f}秒")
        print(f"{'='*60}\n")
        
        return best_params, result_dict
    
    def visualize_result(
        self,
        result_dict: dict,
        output_dir: str
    ):
        """
        可视化配准结果
        
        Args:
            result_dict: 配准结果字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取图像
        us_array = sitk.GetArrayFromImage(self.ultrasound_2d).squeeze()
        slice_array = sitk.GetArrayFromImage(result_dict['best_slice']).squeeze()
        
        # 归一化
        def normalize(img):
            p1, p99 = np.percentile(img[img > -500], [1, 99])
            return np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)
        
        us_norm = normalize(us_array)
        slice_norm = normalize(slice_array)
        
        fig = plt.figure(figsize=(18, 6))
        
        # 1. Ultrasound image
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(us_norm, cmap='gray')
        ax1.set_title('2D Ultrasound', fontsize=14)
        ax1.axis('off')
        
        # 2. Extracted CT slice
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(slice_norm, cmap='gray')
        ax2.set_title('Extracted CT Slice', fontsize=14)
        ax2.axis('off')
        
        # 3. Overlay
        ax3 = plt.subplot(1, 4, 3)
        overlay = np.zeros((*us_norm.shape, 3))
        overlay[:, :, 0] = us_norm  # Red: Ultrasound
        overlay[:, :, 1] = slice_norm  # Green: CT slice
        ax3.imshow(overlay)
        ax3.set_title('Overlay (Red=US, Green=CT)', fontsize=14)
        ax3.axis('off')
        
        # 4. Optimization curve
        ax4 = plt.subplot(1, 4, 4)
        # Support different history formats
        if len(self.optimization_history) > 0:
            if 'chamfer_dist' in self.optimization_history[0]:
                # Powell optimization history
                cost_values = [h['chamfer_dist'] for h in self.optimization_history]
                ax4.plot(cost_values, 'b-', linewidth=2)
                ax4.set_ylabel('Chamfer Distance (mm)', fontsize=12)
            elif 'mean_distance' in self.optimization_history[0]:
                # ICP optimization history
                cost_values = [h['mean_distance'] for h in self.optimization_history]
                ax4.plot(cost_values, 'b-', linewidth=2, marker='o', markersize=6)
                ax4.set_ylabel('Mean Distance (mm)', fontsize=12)
            ax4.set_xlabel('Iteration', fontsize=12)
            ax4.set_title('Optimization Curve', fontsize=14)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        vis_path = output_path / "registration_result.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Visualization saved: {vis_path}")
    
    
    
    def visualize_correspondences(
        self,
        result_dict: dict,
        output_dir: str,
        max_correspondence_dist: float = 50.0,
        show_lines: bool = True,
        subsample_ratio: float = 0.3
    ):
        """
        可视化对应点配准前后的变换
        
        Args:
            result_dict: 配准结果字典（来自register_icp）
            output_dir: 输出目录
            max_correspondence_dist: 最大对应点距离（用于过滤）
            show_lines: 是否显示对应关系连线
            subsample_ratio: 用于显示连线的点采样比例（避免过于密集）
        """
        from scipy.spatial import cKDTree
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 提取边缘点
        us_points = result_dict.get('us_edge_points')
        initial_ct_points = result_dict.get('initial_ct_edge_points')
        final_ct_points = result_dict.get('final_ct_edge_points')
        
        if us_points is None or initial_ct_points is None or final_ct_points is None:
            print("⚠️ Missing edge point data, unable to visualize correspondences")
            return
        
        print(f"\nVisualizing correspondence registration results...")
        print(f"  US edge points: {len(us_points)}")
        print(f"  Initial CT edge points: {len(initial_ct_points)}")
        print(f"  Final CT edge points: {len(final_ct_points)}")
        
        # Build correspondences
        us_tree = cKDTree(us_points)
        
        # Initial correspondences
        initial_distances, initial_indices = us_tree.query(initial_ct_points)
        initial_valid_mask = initial_distances < max_correspondence_dist
        initial_mean_dist = np.mean(initial_distances[initial_valid_mask])
        
        # Final correspondences
        final_distances, final_indices = us_tree.query(final_ct_points)
        final_valid_mask = final_distances < max_correspondence_dist
        final_mean_dist = np.mean(final_distances[final_valid_mask])
        
        print(f"  Initial mean distance: {initial_mean_dist:.3f}mm")
        print(f"  Final mean distance: {final_mean_dist:.3f}mm")
        print(f"  Improvement: {initial_mean_dist - final_mean_dist:.3f}mm")
        
        # Create figure
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Initial state point cloud distribution
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(us_points[:, 0], us_points[:, 1], 
                   c='red', s=20, alpha=0.6, label='US Edge Points')
        ax1.scatter(initial_ct_points[:, 0], initial_ct_points[:, 1], 
                   c='blue', s=20, alpha=0.6, label='Initial CT Edge Points')
        ax1.set_xlabel('X (mm)', fontsize=12)
        ax1.set_ylabel('Y (mm)', fontsize=12)
        ax1.set_title('Initial State: Point Cloud Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 2. Final state point cloud distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(us_points[:, 0], us_points[:, 1], 
                   c='red', s=20, alpha=0.6, label='US Edge Points')
        ax2.scatter(final_ct_points[:, 0], final_ct_points[:, 1], 
                   c='green', s=20, alpha=0.6, label='Final CT Edge Points')
        ax2.set_xlabel('X (mm)', fontsize=12)
        ax2.set_ylabel('Y (mm)', fontsize=12)
        ax2.set_title('Final State: Point Cloud Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # 3. Overlay comparison
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(us_points[:, 0], us_points[:, 1], 
                   c='red', s=20, alpha=0.6, label='US Edge Points', marker='o')
        ax3.scatter(initial_ct_points[:, 0], initial_ct_points[:, 1], 
                   c='blue', s=20, alpha=0.4, label='Initial CT', marker='x')
        ax3.scatter(final_ct_points[:, 0], final_ct_points[:, 1], 
                   c='green', s=20, alpha=0.6, label='Final CT', marker='^')
        ax3.set_xlabel('X (mm)', fontsize=12)
        ax3.set_ylabel('Y (mm)', fontsize=12)
        ax3.set_title('Overlay Comparison', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # 4. Initial correspondences (with lines)
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(us_points[:, 0], us_points[:, 1], 
                   c='red', s=20, alpha=0.6, label='US Edge Points')
        ax4.scatter(initial_ct_points[:, 0], initial_ct_points[:, 1], 
                   c='blue', s=20, alpha=0.6, label='Initial CT Edge Points')
        
        # Add correspondence lines (subsampled to avoid overcrowding)
        if show_lines:
            valid_indices = np.where(initial_valid_mask)[0]
            num_samples = max(1, int(len(valid_indices) * subsample_ratio))
            sampled_indices = np.random.choice(valid_indices, size=num_samples, replace=False)
            
            for idx in sampled_indices:
                ct_pt = initial_ct_points[idx]
                us_pt = us_points[initial_indices[idx]]
                ax4.plot([ct_pt[0], us_pt[0]], [ct_pt[1], us_pt[1]], 
                        'gray', alpha=0.3, linewidth=0.5)
        
        ax4.set_xlabel('X (mm)', fontsize=12)
        ax4.set_ylabel('Y (mm)', fontsize=12)
        ax4.set_title(f'Initial Correspondences (Mean Dist: {initial_mean_dist:.2f}mm)', 
                     fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        # 5. Final correspondences (with lines)
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(us_points[:, 0], us_points[:, 1], 
                   c='red', s=20, alpha=0.6, label='US Edge Points')
        ax5.scatter(final_ct_points[:, 0], final_ct_points[:, 1], 
                   c='green', s=20, alpha=0.6, label='Final CT Edge Points')
        
        # Add correspondence lines
        if show_lines:
            valid_indices = np.where(final_valid_mask)[0]
            num_samples = max(1, int(len(valid_indices) * subsample_ratio))
            sampled_indices = np.random.choice(valid_indices, size=num_samples, replace=False)
            
            for idx in sampled_indices:
                ct_pt = final_ct_points[idx]
                us_pt = us_points[final_indices[idx]]
                ax5.plot([ct_pt[0], us_pt[0]], [ct_pt[1], us_pt[1]], 
                        'gray', alpha=0.3, linewidth=0.5)
        
        ax5.set_xlabel('X (mm)', fontsize=12)
        ax5.set_ylabel('Y (mm)', fontsize=12)
        ax5.set_title(f'Final Correspondences (Mean Dist: {final_mean_dist:.2f}mm)', 
                     fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axis('equal')
        
        # 6. Distance distribution histogram
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(initial_distances[initial_valid_mask], bins=50, alpha=0.5, 
                color='blue', label='Initial Distance', edgecolor='black')
        ax6.hist(final_distances[final_valid_mask], bins=50, alpha=0.5, 
                color='green', label='Final Distance', edgecolor='black')
        ax6.axvline(initial_mean_dist, color='blue', linestyle='--', linewidth=2, 
                   label=f'Initial Mean: {initial_mean_dist:.2f}mm')
        ax6.axvline(final_mean_dist, color='green', linestyle='--', linewidth=2, 
                   label=f'Final Mean: {final_mean_dist:.2f}mm')
        ax6.set_xlabel('Correspondence Distance (mm)', fontsize=12)
        ax6.set_ylabel('Frequency', fontsize=12)
        ax6.set_title('Distance Distribution Comparison', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        corr_path = output_path / "correspondence_visualization.png"
        plt.savefig(corr_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Correspondence visualization saved: {corr_path}")
        
        # Generate statistics report
        stats_path = output_path / "correspondence_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ICP Registration Correspondence Statistics Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"US edge points: {len(us_points)}\n")
            f.write(f"Initial CT edge points: {len(initial_ct_points)}\n")
            f.write(f"Final CT edge points: {len(final_ct_points)}\n\n")
            
            f.write("Initial State:\n")
            f.write(f"  Valid correspondence pairs: {np.sum(initial_valid_mask)}\n")
            f.write(f"  Mean distance: {initial_mean_dist:.3f} mm\n")
            f.write(f"  Min distance: {np.min(initial_distances[initial_valid_mask]):.3f} mm\n")
            f.write(f"  Max distance: {np.max(initial_distances[initial_valid_mask]):.3f} mm\n")
            f.write(f"  Distance std: {np.std(initial_distances[initial_valid_mask]):.3f} mm\n\n")
            
            f.write("Final State:\n")
            f.write(f"  Valid correspondence pairs: {np.sum(final_valid_mask)}\n")
            f.write(f"  Mean distance: {final_mean_dist:.3f} mm\n")
            f.write(f"  Min distance: {np.min(final_distances[final_valid_mask]):.3f} mm\n")
            f.write(f"  Max distance: {np.max(final_distances[final_valid_mask]):.3f} mm\n")
            f.write(f"  Distance std: {np.std(final_distances[final_valid_mask]):.3f} mm\n\n")
            
            f.write("Improvement:\n")
            f.write(f"  Mean distance reduction: {initial_mean_dist - final_mean_dist:.3f} mm\n")
            f.write(f"  Improvement percentage: {(1 - final_mean_dist/initial_mean_dist)*100:.1f}%\n\n")
            
            f.write("Parameter Changes:\n")
            initial_params = result_dict['initial_params']
            best_params = result_dict['best_params']
            param_names = ['α(rad)', 'β(rad)', 'γ(rad)', 'tx(mm)', 'ty(mm)', 'tz(mm)', 'sx', 'sy']
            for i, name in enumerate(param_names):
                f.write(f"  {name}: {initial_params[i]:.4f} → {best_params[i]:.4f} "
                       f"(Δ={best_params[i]-initial_params[i]:.4f})\n")
        
        print(f"✓ Statistics report saved: {stats_path}")


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数"""
    
    # 初始化配准器
    registrator = TwoD_ThreeD_Registration()
   
    # 加载数据（包括mask）
    ct_path = r"D:\dataset\CT\MM-WHS2017\ct_train\ct_train_1004_image.nii"
    ultrasound_path = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_image_initial_transform.nii.gz"
    ct_mask_path = r"D:\dataset\CT\MM-WHS2017\ct_train\ct_train_1004_remapped_label.nii"  # CT心脏mask路径
    us_mask_path = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_label_initial_transform.nii.gz"
    output_dir = r"D:\dataset\TEECT_data\registration_results\2d_3d"
    
    registrator.load_images(ct_path, ultrasound_path, ct_mask_path, us_mask_path)
    # registrator.load_images(ct_path, ultrasound_path)
    
    
    # 从粗略位置开始（假设已经有大致的切面方向）
    # 例如：从2D切片生成器得到的初始参数
    initial_params = np.array([
        np.radians(50),   # alpha: 0°
        np.radians(-25),  # beta: 25°
        np.radians(0),   # gamma: 0°
        5.77, -163.36, 5.0,   # translation: (0, 0, 5) mm
        1.0, 1.0         # scaling: (1.0, 1.0)
    ])
    
    # 执行ICP配准
    best_params, result_dict = registrator.register_icp(
        initial_params=initial_params,
        max_iterations=50,  # ICP外层迭代
        tolerance=0.1,  # 收敛阈值：平均距离变化<0.1mm
        max_correspondence_dist=30.0,  # 最大对应距离30mm
        inner_opt_iterations=50  # 每次ICP迭代内部优化10次
    )
    
    # 可视化结果
    registrator.visualize_result(result_dict, output_dir)
    
    # 可视化对应点配准前后的变换
    registrator.visualize_correspondences(
        result_dict, 
        output_dir,
        max_correspondence_dist=30.0,  # 与配准时使用的相同
        show_lines=True,  # 显示对应关系连线
        subsample_ratio=0.2  # 显示20%的对应连线（避免过于密集）
    )
    
    # 保存最优切片
    best_slice_path = Path(output_dir) / "best_slice_icp.nii.gz"
    sitk.WriteImage(result_dict['best_slice'], str(best_slice_path))
    print(f"✓ 最优切片已保存: {best_slice_path}")


if __name__ == "__main__":
    main()
