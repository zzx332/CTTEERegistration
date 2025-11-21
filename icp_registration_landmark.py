"""
基于边缘标签的ICP配准
从边缘图中提取特征点，使用ICP进行2D刚性配准
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
import pandas as pd
from datetime import datetime


class EdgeBasedICPRegistration:
    """
    基于边缘特征点的ICP配准
    """
    
    def __init__(self):
        """初始化"""
        self.final_transform = None
        self.icp_history = []
        
    def extract_edge_points(
        self,
        edge_image: np.ndarray,
        sampling_method: str = 'uniform',
        num_points: int = 1000,
        min_distance: float = 5.0  # 增大默认最小距离
    ) -> np.ndarray:
        """
        从边缘图中提取特征点
        
        Args:
            edge_image: 边缘图 (H, W)，边缘像素值>0
            sampling_method: 采样方法 ('uniform', 'random', 'curvature', 'all', 'grid')
            num_points: 目标点数（仅用于uniform和random）
            min_distance: 最小点间距（用于uniform采样）
        
        Returns:
            points: Nx2数组，每行是[x, y]坐标
        """
        # 获取所有边缘点
        edge_coords = np.argwhere(edge_image > 0)  # [row, col]
        
        if len(edge_coords) == 0:
            raise ValueError("边缘图中没有边缘点！")
        
        # 转换为[x, y]格式（注意：col=x, row=y）
        all_points = edge_coords[:, [1, 0]].astype(np.float64)
        
        print(f"  边缘图中总共有 {len(all_points)} 个边缘点")
        
        if sampling_method == 'all':
            return all_points
        
        elif sampling_method == 'random':
            # 随机采样
            if len(all_points) <= num_points:
                return all_points
            indices = np.random.choice(len(all_points), num_points, replace=False)
            return all_points[indices]
        
        elif sampling_method == 'grid':
            # 基于网格的均匀采样（推荐）
            sampled_points = self._grid_based_sampling(all_points, num_points, min_distance)
            print(f"  grid采样: {len(sampled_points)} 个点 (min_distance={min_distance})")
            return sampled_points
        
        elif sampling_method == 'uniform':
            # 改进的均匀采样：使用FPS (Farthest Point Sampling)
            sampled_points = self._farthest_point_sampling(all_points, num_points)
            print(f"  uniform采样(FPS): {len(sampled_points)} 个点")
            return sampled_points
        
        elif sampling_method == 'curvature':
            # 基于曲率的采样（提取高曲率点）
            curvature_points = self._extract_high_curvature_points(
                edge_image, all_points, num_points
            )
            return curvature_points
        
        else:
            raise ValueError(f"未知的采样方法: {sampling_method}")

    def _grid_based_sampling(
        self,
        points: np.ndarray,
        num_points: int,
        grid_size: float = 5.0
    ) -> np.ndarray:
        """
        基于网格的均匀采样，确保点均匀分布在整个边缘区域
        
        Args:
            points: 所有边缘点 Nx2
            num_points: 目标采样点数
            grid_size: 网格大小（像素）
        
        Returns:
            sampled_points: 采样后的点
        """
        # 计算点云的边界
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # 计算网格尺寸
        grid_dims = np.ceil((max_coords - min_coords) / grid_size).astype(int) + 1
        
        # 为每个点分配网格索引
        grid_indices = ((points - min_coords) / grid_size).astype(int)
        
        # 创建网格字典，存储每个网格中的点
        grid_dict = {}
        for idx, (gx, gy) in enumerate(grid_indices):
            grid_key = (gx, gy)
            if grid_key not in grid_dict:
                grid_dict[grid_key] = []
            grid_dict[grid_key].append(idx)
        
        print(f"    网格划分: {grid_dims[0]}x{grid_dims[1]} = {len(grid_dict)} 个非空网格")
        
        # 从每个网格中采样点
        sampled_indices = []
        
        # 方法1：每个网格随机选1个点
        for grid_key, point_indices in grid_dict.items():
            # 随机选择一个点
            selected_idx = np.random.choice(point_indices)
            sampled_indices.append(selected_idx)
        
        # 如果点数不够，减小网格尺寸重新采样
        if len(sampled_indices) < max(10, num_points // 2):
            print(f"    ⚠️ 采样点过少({len(sampled_indices)})，减小网格尺寸...")
            return self._grid_based_sampling(points, num_points, grid_size * 0.7)
        
        # 如果点数太多，随机抽取目标数量
        if len(sampled_indices) > num_points:
            sampled_indices = np.random.choice(sampled_indices, num_points, replace=False)
        
        return points[sampled_indices]

    def _farthest_point_sampling(
        self,
        points: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """
        最远点采样(FPS)，保证点之间最大化间距
        
        Args:
            points: 所有边缘点 Nx2
            num_points: 目标采样点数
        
        Returns:
            sampled_points: 采样后的点
        """
        n = len(points)
        if n <= num_points:
            return points
        
        # 随机选择第一个点
        sampled_indices = [np.random.randint(n)]
        distances = np.full(n, np.inf)
        
        for i in range(1, num_points):
            # 更新每个点到已采样点集的最小距离
            last_point = points[sampled_indices[-1]]
            dist_to_last = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, dist_to_last)
            
            # 选择距离最远的点
            farthest_idx = np.argmax(distances)
            sampled_indices.append(farthest_idx)
            
            # 每100个点打印一次进度
            if (i + 1) % 100 == 0:
                print(f"    FPS进度: {i+1}/{num_points}")
        
        return points[sampled_indices]
    
    def _extract_high_curvature_points(
        self,
        edge_image: np.ndarray,
        all_points: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """提取高曲率点（角点、拐点）"""
        from scipy.ndimage import sobel
        
        # 计算梯度
        gx = sobel(edge_image.astype(float), axis=1)
        gy = sobel(edge_image.astype(float), axis=0)
        
        # 计算曲率（梯度变化率）
        gxx = sobel(gx, axis=1)
        gyy = sobel(gy, axis=0)
        curvature = np.abs(gxx) + np.abs(gyy)
        
        # 在边缘点位置提取曲率值
        curvature_values = []
        for point in all_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= y < curvature.shape[0] and 0 <= x < curvature.shape[1]:
                curvature_values.append(curvature[y, x])
            else:
                curvature_values.append(0)
        
        curvature_values = np.array(curvature_values)
        
        # 选择曲率最高的点
        top_indices = np.argsort(curvature_values)[-num_points:]
        return all_points[top_indices]
    
    def icp_2d(
        self,
        fixed_points: np.ndarray,
        moving_points: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 10.0,
        use_point_to_plane: bool = False,
        use_centroid_init: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        2D ICP算法
        
        Args:
            fixed_points: 固定点云 Nx2
            moving_points: 移动点云 Mx2
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值
            max_correspondence_distance: 最大对应距离
            use_point_to_plane: 是否使用point-to-plane（需要法向量）
            use_centroid_init: 是否使用质心对齐作为初始变换
        
        Returns:
            R: 2x2旋转矩阵
            t: 2x1平移向量
            errors: 每次迭代的误差历史
        """
        # 初始化为单位变换（防止循环提前退出时变量未定义）
        R = np.eye(2)
        t = np.zeros(2)
        
        # 应用初始质心对齐
        if use_centroid_init:
            # 计算点云质心
            fixed_centroid = np.mean(fixed_points, axis=0)
            moving_centroid = np.mean(moving_points, axis=0)
            
            # 计算初始平移（将moving质心移动到fixed质心）
            initial_translation = fixed_centroid - moving_centroid
            
            print(f"  初始质心对齐:")
            print(f"    Fixed质心: ({fixed_centroid[0]:.2f}, {fixed_centroid[1]:.2f})")
            print(f"    Moving质心: ({moving_centroid[0]:.2f}, {moving_centroid[1]:.2f})")
            print(f"    初始平移: ({initial_translation[0]:.2f}, {initial_translation[1]:.2f})")
            
            # 应用初始平移到moving点云
            current_points = moving_points + initial_translation
            
            # 更新初始变换
            t = initial_translation
        else:
            current_points = moving_points.copy()
        
        errors = []
        
        # 构建KD树加速最近邻搜索
        tree = cKDTree(fixed_points)
        
        for iteration in range(max_iterations):
            # 1. 找到最近邻对应
            distances, indices = tree.query(current_points)
            
            # 过滤距离过大的对应
            valid_mask = distances < max_correspondence_distance
            if np.sum(valid_mask) < 3:
                print(f"  警告：迭代{iteration}时有效对应点过少 ({np.sum(valid_mask)}个)")
                if iteration == 0:
                    print(f"  提示：初始对齐可能很差，考虑增大max_correspondence_distance或改进初始位置")
                break
            
            valid_moving = current_points[valid_mask]
            valid_fixed = fixed_points[indices[valid_mask]]
            
            # 2. 计算变换（SVD方法）
            R_iter, t_iter = self._compute_rigid_transform_2d(valid_moving, valid_fixed)
            
            # 3. 应用变换
            current_points = (R_iter @ current_points.T).T + t_iter
            
            # 4. 累积变换（R和t是从原始moving_points到当前位置的总变换）
            # 新的总变换 = R_iter * R_old，t_new = R_iter * t_old + t_iter
            t = R_iter @ t + t_iter
            R = R_iter @ R
            
            # 5. 计算误差
            mean_error = np.mean(distances[valid_mask])
            errors.append(mean_error)
            
            # 6. 检查收敛
            if iteration > 0 and abs(errors[-1] - errors[-2]) < tolerance:
                print(f"  ICP收敛于迭代 {iteration}, 误差={mean_error:.4f}")
                break
        
        # 如果没有任何有效迭代，返回初始变换
        if len(errors) == 0:
            print(f"  ✗ ICP失败：没有找到任何有效对应")
            errors.append(float('inf'))
        
        return R, t, errors
    
    def _compute_rigid_transform_2d(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算2D刚性变换（SVD方法）
        
        Args:
            source: 源点云 Nx2
            target: 目标点云 Nx2
        
        Returns:
            R: 2x2旋转矩阵
            t: 2维平移向量
        """
        # 计算质心
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(target, axis=0)
        
        # 去中心化
        source_centered = source - centroid_source
        target_centered = target - centroid_target
        
        # 计算协方差矩阵
        H = source_centered.T @ target_centered
        
        # SVD分解
        U, S, Vt = np.linalg.svd(H)
        
        # 计算旋转矩阵
        R = Vt.T @ U.T
        
        # 确保是旋转矩阵（行列式为1）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 计算平移
        t = centroid_target - R @ centroid_source
        
        return R, t
    
    def register_with_known_correspondences(
        self,
        fixed_landmarks: np.ndarray,
        moving_landmarks: np.ndarray,
        use_scaling: bool = False,
        use_optimization: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[float]]:
        """
        基于已知对应关系的关键点配准
        
        适用场景：
        - 已知固定点和移动点的一一对应关系
        - 解剖关键点配准（如心尖点、瓣环点等）
        - 不需要迭代搜索对应关系
        
        Args:
            fixed_landmarks: 固定图像的关键点 Nx2，按顺序排列
            moving_landmarks: 移动图像的关键点 Nx2，按顺序排列（与fixed_landmarks一一对应）
            use_scaling: 是否使用相似变换（包含缩放）
            use_optimization: 是否使用优化方法（适用于过约束情况）
        
        Returns:
            R: 2x2旋转矩阵（如果use_scaling=True，则包含缩放）
            t: 2维平移向量
            rmse: 配准后的均方根误差
            scale: 缩放因子（仅当use_scaling=True时返回）
        """
        # 检查输入
        if len(fixed_landmarks) != len(moving_landmarks):
            raise ValueError(
                f"固定点和移动点数量必须相同！"
                f"fixed={len(fixed_landmarks)}, moving={len(moving_landmarks)}"
            )
        
        if len(fixed_landmarks) < 2:
            raise ValueError("至少需要2对对应点！")
        
        n_points = len(fixed_landmarks)
        print(f"\n基于已知对应关系的配准")
        print(f"  对应点对数量: {n_points}")
        print(f"  使用缩放: {use_scaling}")
        print(f"  使用优化: {use_optimization}")
        
        if not use_optimization:
            # 方法1：闭式解（SVD方法）
            if use_scaling:
                R, t, scale = self._compute_similarity_transform_2d(
                    moving_landmarks, fixed_landmarks
                )
                print(f"  缩放因子: {scale:.4f}")
            else:
                R, t = self._compute_rigid_transform_2d(
                    moving_landmarks, fixed_landmarks
                )
                scale = None
        else:
            # 方法2：非线性优化（适用于过约束情况，点数>3）
            from scipy.optimize import minimize
            
            def objective(params):
                """目标函数：最小化对应点对之间的距离"""
                if use_scaling:
                    angle, tx, ty, s = params
                else:
                    angle, tx, ty = params
                    s = 1.0
                
                # 构建变换矩阵
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                R_opt = s * np.array([[cos_a, -sin_a],
                                       [sin_a, cos_a]])
                t_opt = np.array([tx, ty])
                
                # 应用变换
                transformed = (R_opt @ moving_landmarks.T).T + t_opt
                
                # 计算误差
                errors = np.linalg.norm(transformed - fixed_landmarks, axis=1)
                return np.sum(errors ** 2)  # 最小化平方误差和
            
            # 使用SVD解作为初始值
            if use_scaling:
                R_init, t_init, scale_init = self._compute_similarity_transform_2d(
                    moving_landmarks, fixed_landmarks
                )
                angle_init = np.arctan2(R_init[1, 0], R_init[0, 0])
                x0 = [angle_init, t_init[0], t_init[1], scale_init]
            else:
                R_init, t_init = self._compute_rigid_transform_2d(
                    moving_landmarks, fixed_landmarks
                )
                angle_init = np.arctan2(R_init[1, 0], R_init[0, 0])
                x0 = [angle_init, t_init[0], t_init[1]]
            
            print(f"  优化初始值: angle={np.degrees(x0[0]):.2f}°, t=({x0[1]:.2f}, {x0[2]:.2f})")
            
            # 优化
            result = minimize(objective, x0, method='L-BFGS-B')
            
            if result.success:
                print(f"  ✓ 优化成功，迭代次数: {result.nit}")
            else:
                print(f"  ⚠️ 优化未完全收敛: {result.message}")
            
            # 提取优化后的参数
            if use_scaling:
                angle, tx, ty, scale = result.x
            else:
                angle, tx, ty = result.x
                scale = None
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            s = scale if use_scaling else 1.0
            R = s * np.array([[cos_a, -sin_a],
                              [sin_a, cos_a]])
            t = np.array([tx, ty])
        
        # 计算配准后的RMSE
        transformed = (R @ moving_landmarks.T).T + t
        errors = np.linalg.norm(transformed - fixed_landmarks, axis=1)
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # 提取旋转角度
        angle = np.arctan2(R[1, 0], R[0, 0])
        angle_deg = np.degrees(angle)
        
        print(f"\n  配准结果:")
        print(f"    旋转角度: {angle_deg:.2f}°")
        print(f"    平移: ({t[0]:.2f}, {t[1]:.2f}) 像素")
        if scale is not None:
            print(f"    缩放: {scale:.4f}")
        print(f"    RMSE: {rmse:.4f} 像素")
        print(f"    各点误差: {errors}")
        
        return R, t, rmse, scale


    def _compute_similarity_transform_2d(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        计算2D相似变换（旋转+平移+缩放）
        
        使用Umeyama算法（闭式解）
        
        Args:
            source: 源点云 Nx2
            target: 目标点云 Nx2
        
        Returns:
            R: 2x2旋转矩阵（不含缩放）
            t: 2维平移向量
            scale: 缩放因子
        """
        # 计算质心
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(target, axis=0)
        
        # 去中心化
        source_centered = source - centroid_source
        target_centered = target - centroid_target
        
        # 计算缩放因子
        var_source = np.mean(np.sum(source_centered ** 2, axis=1))
        
        # 计算协方差矩阵
        H = source_centered.T @ target_centered / len(source)
        
        # SVD分解
        U, S, Vt = np.linalg.svd(H)
        
        # 计算旋转矩阵
        R = Vt.T @ U.T
        
        # 确保是旋转矩阵（行列式为1）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 计算缩放因子
        scale = np.trace(np.diag(S)) / var_source
        
        # 计算平移（考虑缩放）
        t = centroid_target - scale * R @ centroid_source
        
        # 注意：返回的R不包含缩放，需要在使用时乘以scale
        return R, t, scale


    def register(
        self,
        fixed_edge_path: str,
        moving_edge_path: str,
        output_dir: str = None,
        sampling_method: str = 'uniform',
        num_points: int = 500,
        max_iterations: int = 100,
        max_correspondence_distance: float = 10.0,
        visualize: bool = True,
        use_known_correspondences: bool = False,  # 新增参数
        fixed_landmarks: Optional[np.ndarray] = None,  # 新增参数
        moving_landmarks: Optional[np.ndarray] = None,  # 新增参数
        use_scaling: bool = False  # 新增参数
    ) -> Tuple[sitk.Transform, dict]:
        """
        执行基于边缘的ICP配准或基于已知对应关系的配准
        
        Args:
            fixed_edge_path: 固定图像的边缘图路径
            moving_edge_path: 移动图像的边缘图路径
            output_dir: 输出目录
            sampling_method: 特征点采样方法
            num_points: 采样点数
            max_iterations: ICP最大迭代次数
            max_correspondence_distance: 最大对应距离
            visualize: 是否可视化
            use_known_correspondences: 是否使用已知对应关系（不使用ICP）
            fixed_landmarks: 固定图像的关键点 Nx2（当use_known_correspondences=True时必需）
            moving_landmarks: 移动图像的关键点 Nx2（当use_known_correspondences=True时必需）
            use_scaling: 是否使用相似变换（包含缩放）
        
        Returns:
            transform: SimpleITK变换对象
            metrics: 配准指标字典
        """
        print(f"\n{'='*60}")
        if use_known_correspondences:
            print(f"开始基于已知对应关系的配准")
        else:
            print(f"开始ICP配准")
        print(f"固定图像: {Path(fixed_edge_path).name}")
        print(f"移动图像: {Path(moving_edge_path).name}")
        print(f"{'='*60}\n")
        
        # 读取边缘图
        fixed_img = sitk.ReadImage(fixed_edge_path)
        moving_img = sitk.ReadImage(moving_edge_path)
        print("  转换超声图像坐标系...")
        fixed_img = sitk.Flip(fixed_img, flipAxes=[True, True])
        fixed_img.SetOrigin((0,0))
        print("    已转换为左手坐标系（X轴翻转, Y轴翻转）")
        fixed_array = sitk.GetArrayFromImage(fixed_img).squeeze()
        moving_array = sitk.GetArrayFromImage(moving_img).squeeze()
        
        print(f"固定图像形状: {fixed_array.shape}")
        print(f"移动图像形状: {moving_array.shape}")
        
        if use_known_correspondences:
            # 使用已知对应关系
            if fixed_landmarks is None or moving_landmarks is None:
                raise ValueError("use_known_correspondences=True时必须提供fixed_landmarks和moving_landmarks！")
            fixed_landmarks_phy = self._transform_points_to_physical(fixed_landmarks, fixed_img)
            moving_landmarks_phy = self._transform_points_to_physical(moving_landmarks, moving_img)
            print(f"\n使用已知对应关系进行配准...")
            R, t, rmse, scale = self.register_with_known_correspondences(
                fixed_landmarks_phy,
                moving_landmarks_phy,
                use_scaling=use_scaling,
                use_optimization=True  # 使用优化方法
            )
            
            errors = [rmse]  # 用于可视化
            fixed_points = fixed_landmarks
            moving_points = moving_landmarks
            
        else:
            # 使用ICP
            # 提取特征点
            print(f"\n提取特征点 (方法={sampling_method}, 目标点数={num_points})...")
            fixed_points = self.extract_edge_points(
                fixed_array, sampling_method, num_points
            )
            moving_points = self.extract_edge_points(
                moving_array, sampling_method, num_points
            )
            print(f"  固定图像特征点: {len(fixed_points)}")
            print(f"  移动图像特征点: {len(moving_points)}")
            
            # 执行ICP
            print(f"\n执行ICP配准 (最大迭代={max_iterations})...")
            R, t, errors = self.icp_2d(
                fixed_points,
                moving_points,
                max_iterations=max_iterations,
                max_correspondence_distance=max_correspondence_distance
            )
            scale = None
        
        # 转换为SimpleITK变换（使用逆变换）
        R_inv = np.linalg.inv(R)
        t_inv = -R_inv @ t
        angle_inv = np.arctan2(R_inv[1, 0], R_inv[0, 0])
        
        if use_scaling and scale is not None:
            # 使用Similarity2DTransform
            transform = sitk.Similarity2DTransform()
            transform.SetAngle(angle_inv)
            transform.SetTranslation(t_inv.tolist())
            transform.SetScale(1.0 / scale)  # 逆变换的缩放是倒数
        else:
            # 使用AffineTransform
            transform = sitk.AffineTransform(2)
            transform.SetMatrix(R_inv.flatten().tolist())
            transform.SetTranslation(t_inv.tolist())
        
        self.final_transform = transform
        self.icp_history = errors
        
        # 准备指标（使用前向变换的参数）
        angle = np.arctan2(R[1, 0], R[0, 0])
        angle_deg = np.degrees(angle)
        
        metrics = {
            'rotation_angle_deg': angle_deg,
            'translation_x': t[0],
            'translation_y': t[1],
            'final_error': errors[-1],
            'num_iterations': len(errors) if not use_known_correspondences else 1,
            'num_fixed_points': len(fixed_points),
            'num_moving_points': len(moving_points),
            'method': 'known_correspondences' if use_known_correspondences else 'icp'
        }
        
        if scale is not None:
            metrics['scale'] = scale
        
        # ========== 新增：保存变换 ==========
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存变换文件（.tfm格式）
            transform_path = output_dir / f"{Path(moving_edge_path).stem}_transform.tfm"
            sitk.WriteTransform(transform, str(transform_path))
            print(f"\n✓ 变换已保存: {transform_path}")
            
        # ========== 保存变换结束 ==========
                # 应用变换
        print("\n5. 应用变换...")
        resampled_image = sitk.Resample(
            moving_img,
            fixed_img,
            transform,
            sitk.sitkNearestNeighbor,
            0.0,
            moving_img.GetPixelID()
        )
        sitk.WriteImage(resampled_image, output_dir / f"{Path(moving_edge_path).stem}_resampled.nii.gz")
        print(f"  已保存: {output_dir / f'{Path(moving_edge_path).stem}_resampled.nii.gz'}")
        # 可视化
        if visualize:
            # 如果使用物理坐标，转换为像素坐标用于显示点的位置
            self._visualize_registration(
                fixed_img, moving_img, resampled_image,
                fixed_points, moving_points,
                R, t, errors,
                output_dir, Path(moving_edge_path).stem
            )
        
        return transform, metrics

    def _transform_points_to_physical(self, points: np.ndarray, img: sitk.Image) -> np.ndarray:
        landmarks_phy = []
        for landmark in points:
            idx = tuple(int(x) for x in landmark)
            landmark_phy = img.TransformIndexToPhysicalPoint(idx)
            landmarks_phy.append(landmark_phy)
        return np.array(landmarks_phy)

    def _visualize_registration(
        self,
        fixed_img: sitk.Image,
        moving_img: sitk.Image,
        resampled_image: sitk.Image,
        fixed_points: np.ndarray,
        moving_points: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        errors: List[float],
        output_dir: Path,
        base_name: str
    ):
        """可视化配准结果"""
        
        fixed_array = sitk.GetArrayFromImage(fixed_img).squeeze()
        moving_array = sitk.GetArrayFromImage(moving_img).squeeze()
        resampled_array = sitk.GetArrayFromImage(resampled_image).squeeze()
        # 应用变换到移动点
        
        fig = plt.figure(figsize=(20, 10))
        
        # 1. 特征点提取
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(fixed_array, cmap='gray')
        ax1.scatter(fixed_points[:, 0], fixed_points[:, 1], 
                   c='red', s=10, alpha=0.5, label='Fixed Points')
        ax1.set_title(f'Fixed Edge Points\n({len(fixed_points)} points)')
        ax1.legend()
        ax1.axis('off')
        
        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(moving_array, cmap='gray')
        ax2.scatter(moving_points[:, 0], moving_points[:, 1], 
                   c='blue', s=10, alpha=0.5, label='Moving Points')
        ax2.set_title(f'Moving Edge Points\n({len(moving_points)} points)')
        ax2.legend()
        ax2.axis('off')
        
        # 2. 配准前点云叠加
        fixed_points_phy = self._transform_points_to_physical(fixed_points, fixed_img)
        moving_points_phy = self._transform_points_to_physical(moving_points, moving_img)
        ax3 = plt.subplot(2, 4, 3)
        ax3.scatter(fixed_points_phy[:, 0], fixed_points_phy[:, 1], 
                   c='red', s=20, alpha=0.6, label='Fixed')
        ax3.scatter(moving_points_phy[:, 0], moving_points_phy[:, 1], 
                   c='blue', s=20, alpha=0.6, label='Moving')
        ax3.set_title('Before Registration')
        ax3.legend()
        ax3.axis('equal')
        ax3.grid(True, alpha=0.3)
        
        # 3. 配准后点云叠加
        transformed_points_phy = (R @ moving_points_phy.T).T + t
        ax4 = plt.subplot(2, 4, 4)
        ax4.scatter(fixed_points_phy[:, 0], fixed_points_phy[:, 1], 
                   c='red', s=20, alpha=0.6, label='Fixed')
        ax4.scatter(transformed_points_phy[:, 0], transformed_points_phy[:, 1], 
                   c='green', s=20, alpha=0.6, label='Transformed')
        ax4.set_title('After Registration')
        ax4.legend()
        ax4.axis('equal')
        ax4.grid(True, alpha=0.3)
        
        # 4. ICP收敛曲线
        ax5 = plt.subplot(2, 4, 5)
        ax5.plot(errors, 'b-', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Mean Error (pixels)')
        ax5.set_title('ICP Convergence')
        ax5.grid(True, alpha=0.3)
        
        # 5. 对应关系可视化（采样显示）
        ax6 = plt.subplot(2, 4, 6)
        sample_indices = np.random.choice(len(fixed_points), 
                                         min(50, len(fixed_points)), 
                                         replace=False)
        
        # # 找到最近邻
        # from scipy.spatial import cKDTree
        # tree = cKDTree(fixed_points)
        # _, indices = tree.query(transformed_points)
        
        # for idx in sample_indices:
        #     moving_idx = np.where(indices == idx)[0]
        #     if len(moving_idx) > 0:
        #         moving_idx = moving_idx[0]
        #         ax6.plot([fixed_points[idx, 0], transformed_points[moving_idx, 0]],
        #                 [fixed_points[idx, 1], transformed_points[moving_idx, 1]],
        #                 'g-', alpha=0.3, linewidth=0.5)
        for idx in sample_indices:
            ax6.plot([fixed_points_phy[idx, 0], transformed_points_phy[idx, 0]],
                        [fixed_points_phy[idx, 1], transformed_points_phy[idx, 1]],
                        'g-', alpha=0.3, linewidth=0.5)
        ax6.scatter(fixed_points_phy[:, 0], fixed_points_phy[:, 1], 
                   c='red', s=10, alpha=0.6)
        ax6.scatter(transformed_points_phy[:, 0], transformed_points_phy[:, 1], 
                   c='green', s=10, alpha=0.6)
        ax6.set_title('Point Correspondences\n(sampled)')
        ax6.axis('equal')
        ax6.grid(True, alpha=0.3)
        
        # ========== 修复部分开始 ==========
        
        # 6. 边缘图叠加（配准前）- 需要先重采样moving_img
        ax7 = plt.subplot(2, 4, 7)
        
        # 将moving_img重采样到fixed_img
        identity_transform = sitk.Transform(2, sitk.sitkIdentity)
        moving_resampled_for_comparison = sitk.Resample(
            moving_img,
            fixed_img,
            identity_transform,
            sitk.sitkNearestNeighbor,
            0.0,
            moving_img.GetPixelID()
        )
        moving_array_resized = sitk.GetArrayFromImage(moving_resampled_for_comparison).squeeze()
        
        overlay_before = np.zeros((*fixed_array.shape, 3))
        overlay_before[:, :, 0] = (fixed_array > 0) * 0.5
        overlay_before[:, :, 2] = (moving_array_resized > 0) * 0.5
        ax7.imshow(overlay_before)
        ax7.set_title('Edge Overlay - Before\n(Red: Fixed, Blue: Moving)')
        ax7.axis('off')
        
        # 7. 边缘图叠加（配准后）
        ax8 = plt.subplot(2, 4, 8)
        resampled_array = sitk.GetArrayFromImage(resampled_image)

        overlay_after = np.zeros((*fixed_array.shape, 3))
        overlay_after[:, :, 0] = (fixed_array > 0) * 0.5
        overlay_after[:, :, 1] = (resampled_array > 0) * 0.5
        ax8.imshow(overlay_after)
        ax8.set_title('Edge Overlay - After\n(Red: Fixed, Green: Transformed)')
        ax8.axis('off')
        
        # ========== 修复部分结束 ==========
        
        plt.tight_layout()
        
        vis_path = output_dir / f"{base_name}_icp_registration.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ 可视化已保存: {vis_path}")
    
    def batch_register(
        self,
        fixed_edge_path: str,
        moving_edge_dir: str,
        output_dir: str,
        **kwargs
    ):
        """批量配准"""
        moving_edge_dir = Path(moving_edge_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        moving_files = sorted(moving_edge_dir.glob("*.nii*"))
        
        print(f"\n找到 {len(moving_files)} 个移动图像")
        print(f"固定图像: {Path(fixed_edge_path).name}")
        print(f"{'='*60}\n")
        
        results = []
        
        for moving_path in moving_files:
            try:
                transform, metrics = self.register(
                    fixed_edge_path,
                    str(moving_path),
                    str(output_dir),
                    **kwargs
                )
                
                metrics['moving_image'] = moving_path.name
                metrics['success'] = True
                results.append(metrics)
                
            except Exception as e:
                print(f"✗ 配准失败: {moving_path.name}")
                print(f"  错误: {e}")
                results.append({
                    'moving_image': moving_path.name,
                    'success': False,
                    'error': str(e)
                })
        
        # 保存结果
        df = pd.DataFrame(results)
        csv_path = output_dir / "icp_registration_results.csv"
        df.to_csv(csv_path, index=False)
        
        excel_path = output_dir / "icp_registration_results.xlsx"
        df.to_excel(excel_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ 批量配准完成！")
        print(f"成功: {df['success'].sum()} / {len(df)}")
        print(f"结果已保存:")
        print(f"  - {csv_path}")
        print(f"  - {excel_path}")


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数"""
    
    # 初始化ICP配准器
    icp_reg = EdgeBasedICPRegistration()
    
    # 单个配准示例
    moving_edge = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge\slice_062_t0.0_rx0_ry0_edge.nii.gz"
    fixed_edge = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_label_edge.nii.gz"
    output_dir = r"D:\dataset\TEECT_data\registration_results\icp"
    moving_img = sitk.ReadImage(moving_edge)
    fixed_img = sitk.Flip(sitk.ReadImage(fixed_edge), flipAxes=[True, True])
    fixed_img.SetOrigin((0,0))
    fixed_landmarks = np.array([[433, 177], [377,168], [333,333], [385,337], [425, 360], [484,343]])
    moving_landmarks = np.array([[297, 188], [283,166], [165,205], [205,232], [217, 250], [259,279]])


    transform, metrics = icp_reg.register(
        fixed_edge_path=fixed_edge,
        moving_edge_path=moving_edge,
        output_dir=output_dir,
        sampling_method='grid',      # 'uniform', 'random', 'curvature', 'all', 'grid'
        num_points=500,                 # 采样点数
        max_iterations=100,             # ICP最大迭代次数
        max_correspondence_distance=10.0,  # 最大对应距离（像素）
        visualize=True,
        use_known_correspondences=True,
        fixed_landmarks=fixed_landmarks,
        moving_landmarks=moving_landmarks,
        use_scaling=True
    )

    # resampled = sitk.Resample(
    #     moving_img,
    #     fixed_img,
    #     transform,
    #     sitk.sitkNearestNeighbor,
    #     0.0
    # )
    # sitk.WriteImage(resampled, r"D:\dataset\TEECT_data\registration_results\icp\slice_062_t0.0_rx0_ry0_edge_resampled.nii.gz")
    # transform, metrics = icp_reg.register(
    #     fixed_edge_path=fixed_edge,
    #     moving_edge_path=moving_edge,
    #     output_dir=output_dir,
    #     sampling_method='grid',      # 'uniform', 'random', 'curvature', 'all', 'grid'
    #     num_points=500,                 # 采样点数
    #     max_iterations=100,             # ICP最大迭代次数
    #     max_correspondence_distance=10.0,  # 最大对应距离（像素）
    #     visualize=True
    # )
    
    print(f"\n配准指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def batch_registration_example():
    """批量配准示例"""
    
    icp_reg = EdgeBasedICPRegistration()
    
    moving_edge_dir = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge_individual\slice_062_t0.0_rx0_ry0_multi_edge.nii.gz"
    # fixed_edge = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge_individual\slice_062_t0.0_rx0_ry0_multi_edge.nii.gz"
    fixed_edge = r"D:\dataset\TEECT_data\tee\tee_train_1004_edge"
    # moving_edge_dir = r"D:\dataset\TEECT_data\tee\tee_train_1004_edge"
    output_dir = r"D:\dataset\TEECT_data\registration_results\icp_batch"
    
    icp_reg.batch_register(
        fixed_edge_path=fixed_edge,
        moving_edge_dir=moving_edge_dir,
        output_dir=output_dir,
        sampling_method='grid',
        num_points=500,
        max_iterations=100,
        max_correspondence_distance=10.0,
        visualize=True
    )


def landmark_registration_example():
    """基于已知对应关系的配准示例"""
    
    icp_reg = EdgeBasedICPRegistration()
    
    # 定义对应的关键点（按顺序一一对应）
    # 例如：心尖点、左心房顶点、右心房顶点、二尖瓣环点、三尖瓣环点等
    fixed_landmarks = np.array([
        [433, 177],  # 点1：心尖点
        [377, 168],  # 点2：左心房顶点
        [333, 333],  # 点3：右心房顶点
        [385, 337],  # 点4：二尖瓣环点
        [425, 360],  # 点5：三尖瓣环点
        [484, 343]   # 点6：其他关键点
    ])
    
    moving_landmarks = np.array([
        [297, 188],  # 点1：心尖点（对应fixed的点1）
        [283, 166],  # 点2：左心房顶点（对应fixed的点2）
        [165, 205],  # 点3：右心房顶点（对应fixed的点3）
        [205, 232],  # 点4：二尖瓣环点（对应fixed的点4）
        [217, 250],  # 点5：三尖瓣环点（对应fixed的点5）
        [259, 279]   # 点6：其他关键点（对应fixed的点6）
    ])
    
    moving_edge = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge\slice_062_t0.0_rx0_ry0_edge.nii.gz"
    fixed_edge = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_label_edge.nii.gz"
    output_dir = r"D:\dataset\TEECT_data\registration_results\landmark"
    
    transform, metrics = icp_reg.register(
        fixed_edge_path=fixed_edge,
        moving_edge_path=moving_edge,
        output_dir=output_dir,
        use_known_correspondences=True,  # 使用已知对应关系
        fixed_landmarks=fixed_landmarks,
        moving_landmarks=moving_landmarks,
        use_scaling=False,  # 是否使用缩放
        visualize=True
    )
    
    print(f"\n配准指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # 运行单个配准
    main()
    
    # 或运行批量配准
    # batch_registration_example()
    
    # 或运行基于已知对应关系的配准
    # landmark_registration_example()