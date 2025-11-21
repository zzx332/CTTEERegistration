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
        params_real[:6] = params_norm[:6] * self.param_scales[:6]
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
        initial_params_norm[:6] = initial_params[:6] / self.param_scales[:6]
        
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
            best_params[:6] = best_params_norm[:6] * self.param_scales[:6]
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
        
        # 1. 超声图像
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(us_norm, cmap='gray')
        ax1.set_title('2D Ultrasound', fontsize=14)
        ax1.axis('off')
        
        # 2. 提取的CT切片
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(slice_norm, cmap='gray')
        ax2.set_title('Extracted CT Slice', fontsize=14)
        ax2.axis('off')
        
        # 3. 叠加图
        ax3 = plt.subplot(1, 4, 3)
        overlay = np.zeros((*us_norm.shape, 3))
        overlay[:, :, 0] = us_norm  # 红色：超声
        overlay[:, :, 1] = slice_norm  # 绿色：CT切片
        ax3.imshow(overlay)
        ax3.set_title('Overlay (Red=US, Green=CT)', fontsize=14)
        ax3.axis('off')
        
        # 4. 优化曲线
        ax4 = plt.subplot(1, 4, 4)
        chamfer_values = [h['chamfer_dist'] for h in self.optimization_history]
        ax4.plot(chamfer_values, 'b-', linewidth=2)
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Chamfer Distance', fontsize=12)
        ax4.set_title('Optimization Curve', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        vis_path = output_path / "registration_result.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ 可视化已保存: {vis_path}")


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
    
    
    # 从粗略位置开始（假设已经有大致的切面方向）
    # 例如：从2D切片生成器得到的初始参数
    initial_params = np.array([
        np.radians(50),   # alpha: 0°
        np.radians(-25),  # beta: 25°
        np.radians(0),   # gamma: 0°
        5.77, -163.36, 5.0,   # translation: (0, 0, 5) mm
        1.0, 1.0         # scaling: (1.0, 1.0)
    ])
    
    # 执行配准
    best_params, result_dict = registrator.register(
        initial_params=initial_params,
        method='powell',  # 或 'de' for global search
        max_iterations=200
    )
    
    # 可视化结果
    registrator.visualize_result(result_dict, output_dir)
    
    # 保存最优切片
    best_slice_path = Path(output_dir) / "best_slice.nii.gz"
    sitk.WriteImage(result_dict['best_slice'], str(best_slice_path))
    print(f"✓ 最优切片已保存: {best_slice_path}")


if __name__ == "__main__":
    main()
