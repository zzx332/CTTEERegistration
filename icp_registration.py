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
    
    def register(
        self,
        fixed_edge_path: str,
        moving_edge_path: str,
        output_dir: str = None,
        sampling_method: str = 'uniform',
        num_points: int = 500,
        max_iterations: int = 100,
        max_correspondence_distance: float = 10.0,
        visualize: bool = True
    ) -> Tuple[sitk.Transform, dict]:
        """
        执行基于边缘的ICP配准
        
        Args:
            fixed_edge_path: 固定图像的边缘图路径
            moving_edge_path: 移动图像的边缘图路径
            output_dir: 输出目录
            sampling_method: 特征点采样方法
            num_points: 采样点数
            max_iterations: ICP最大迭代次数
            max_correspondence_distance: 最大对应距离
            visualize: 是否可视化
        
        Returns:
            transform: SimpleITK变换对象
            metrics: 配准指标字典
        """
        print(f"\n{'='*60}")
        print(f"开始ICP配准")
        print(f"固定图像: {Path(fixed_edge_path).name}")
        print(f"移动图像: {Path(moving_edge_path).name}")
        print(f"{'='*60}\n")
        
        # 读取边缘图
        fixed_img = sitk.ReadImage(fixed_edge_path)
        moving_img = sitk.ReadImage(moving_edge_path)
        print("  转换超声图像坐标系...")
        fixed_img = sitk.Flip(fixed_img, flipAxes=[True, True])  # X轴和Y轴都翻转
        fixed_img.SetOrigin((0,0))
        print("    已转换为左手坐标系（X轴翻转, Y轴翻转）")
        fixed_array = sitk.GetArrayFromImage(fixed_img).squeeze()
        moving_array = sitk.GetArrayFromImage(moving_img).squeeze()
        
        print(f"固定图像形状: {fixed_array.shape}")
        print(f"移动图像形状: {moving_array.shape}")
        
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
        
        # 转换为SimpleITK变换（使用逆变换）
        # ICP计算的是前向变换: p_fixed = R @ p_moving + t
        # SimpleITK的Resample需要逆变换: p_moving = R^(-1) @ (p_fixed - t)

        R_inv = np.linalg.inv(R)
        t_inv = -R_inv @ t

        # 提取逆变换的角度
        angle_inv = np.arctan2(R_inv[1, 0], R_inv[0, 0])

        # 使用AffineTransform保存逆变换
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
            'num_iterations': len(errors),
            'num_fixed_points': len(fixed_points),
            'num_moving_points': len(moving_points)
        }
        
        # 可视化
        if visualize and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self._visualize_registration(
                fixed_array, moving_array,
                fixed_points, moving_points,
                R, t, errors,
                output_dir, Path(moving_edge_path).stem
            )
        
        return transform, metrics
    
    def _visualize_registration(
        self,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        fixed_points: np.ndarray,
        moving_points: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        errors: List[float],
        output_dir: Path,
        base_name: str
    ):
        """可视化配准结果"""
        
        # 应用变换到移动点
        transformed_points = (R @ moving_points.T).T + t
        
        fig = plt.figure(figsize=(20, 10))
        
        # 1. 特征点提取
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(fixed_img, cmap='gray')
        ax1.scatter(fixed_points[:, 0], fixed_points[:, 1], 
                   c='red', s=10, alpha=0.5, label='Fixed Points')
        ax1.set_title(f'Fixed Edge Points\n({len(fixed_points)} points)')
        ax1.legend()
        ax1.axis('off')
        
        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(moving_img, cmap='gray')
        ax2.scatter(moving_points[:, 0], moving_points[:, 1], 
                   c='blue', s=10, alpha=0.5, label='Moving Points')
        ax2.set_title(f'Moving Edge Points\n({len(moving_points)} points)')
        ax2.legend()
        ax2.axis('off')
        
        # 2. 配准前点云叠加
        ax3 = plt.subplot(2, 4, 3)
        ax3.scatter(fixed_points[:, 0], fixed_points[:, 1], 
                   c='red', s=20, alpha=0.6, label='Fixed')
        ax3.scatter(moving_points[:, 0], moving_points[:, 1], 
                   c='blue', s=20, alpha=0.6, label='Moving')
        ax3.set_title('Before Registration')
        ax3.legend()
        ax3.axis('equal')
        ax3.grid(True, alpha=0.3)
        
        # 3. 配准后点云叠加
        ax4 = plt.subplot(2, 4, 4)
        ax4.scatter(fixed_points[:, 0], fixed_points[:, 1], 
                   c='red', s=20, alpha=0.6, label='Fixed')
        ax4.scatter(transformed_points[:, 0], transformed_points[:, 1], 
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
        
        # 找到最近邻
        from scipy.spatial import cKDTree
        tree = cKDTree(fixed_points)
        _, indices = tree.query(transformed_points)
        
        for idx in sample_indices:
            moving_idx = np.where(indices == idx)[0]
            if len(moving_idx) > 0:
                moving_idx = moving_idx[0]
                ax6.plot([fixed_points[idx, 0], transformed_points[moving_idx, 0]],
                        [fixed_points[idx, 1], transformed_points[moving_idx, 1]],
                        'g-', alpha=0.3, linewidth=0.5)
        
        ax6.scatter(fixed_points[:, 0], fixed_points[:, 1], 
                   c='red', s=10, alpha=0.6)
        ax6.scatter(transformed_points[:, 0], transformed_points[:, 1], 
                   c='green', s=10, alpha=0.6)
        ax6.set_title('Point Correspondences\n(sampled)')
        ax6.axis('equal')
        ax6.grid(True, alpha=0.3)
        
        # ========== 修复部分开始 ==========
        
        # 6. 边缘图叠加（配准前）- 需要先重采样moving_img
        ax7 = plt.subplot(2, 4, 7)
        
        # 将moving_img重采样到fixed_img的尺寸（用于可视化）
        if fixed_img.shape != moving_img.shape:
            # 使用SimpleITK进行重采样
            moving_sitk_orig = sitk.GetImageFromArray(moving_img)
            fixed_sitk_ref = sitk.GetImageFromArray(fixed_img)
            
            # 使用单位变换（不进行配准，只是改变尺寸）
            identity_transform = sitk.Transform(2, sitk.sitkIdentity)
            
            moving_resampled_before = sitk.Resample(
                moving_sitk_orig,
                fixed_sitk_ref,  # 参考图像（提供目标尺寸）
                identity_transform,
                sitk.sitkNearestNeighbor,  # 边缘图用最近邻
                0.0
            )
            moving_img_resampled = sitk.GetArrayFromImage(moving_resampled_before)
        else:
            moving_img_resampled = moving_img
        
        overlay_before = np.zeros((*fixed_img.shape, 3))
        overlay_before[:, :, 0] = (fixed_img > 0) * 0.5
        overlay_before[:, :, 2] = (moving_img_resampled > 0) * 0.5
        ax7.imshow(overlay_before)
        ax7.set_title('Edge Overlay - Before\n(Red: Fixed, Blue: Moving)')
        ax7.axis('off')
        
        # 7. 边缘图叠加（配准后）
        ax8 = plt.subplot(2, 4, 8)

        # 创建SimpleITK图像
        moving_sitk = sitk.GetImageFromArray(moving_img)
        fixed_sitk = sitk.GetImageFromArray(fixed_img)

        # ===== 关键：SimpleITK的Resample使用逆变换！=====
        # ICP计算的是前向变换: p_fixed = R @ p_moving + t
        # 但Resample需要逆变换: p_moving = R^(-1) @ (p_fixed - t)

        # 计算逆变换
        R_inv = np.linalg.inv(R)
        t_inv = -R_inv @ t

        print(f"\n  应用逆变换到图像:")
        print(f"    前向旋转矩阵:\n{R}")
        print(f"    前向平移: {t}")
        print(f"    逆旋转矩阵:\n{R_inv}")
        print(f"    逆平移: {t_inv}")

        # 使用AffineTransform设置逆变换
        transform = sitk.AffineTransform(2)
        transform.SetMatrix(R_inv.flatten().tolist())
        transform.SetTranslation(t_inv.tolist())

        # 重采样
        resampled = sitk.Resample(
            moving_sitk, 
            fixed_sitk,
            transform,
            sitk.sitkNearestNeighbor,
            0.0
        )
        resampled_array = sitk.GetArrayFromImage(resampled)

        overlay_after = np.zeros((*fixed_img.shape, 3))
        overlay_after[:, :, 0] = (fixed_img > 0) * 0.5
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
    
    transform, metrics = icp_reg.register(
        fixed_edge_path=fixed_edge,
        moving_edge_path=moving_edge,
        output_dir=output_dir,
        sampling_method='grid',      # 'uniform', 'random', 'curvature', 'all', 'grid'
        num_points=500,                 # 采样点数
        max_iterations=100,             # ICP最大迭代次数
        max_correspondence_distance=10.0,  # 最大对应距离（像素）
        visualize=True
    )
    
    print(f"\n配准指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def batch_registration_example():
    """批量配准示例"""
    
    icp_reg = EdgeBasedICPRegistration()
    
    fixed_edge = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge_individual\slice_062_t0.0_rx0_ry0_multi_edge.nii.gz"
    moving_edge_dir = r"D:\dataset\TEECT_data\tee\tee_train_1004_edge"
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


if __name__ == "__main__":
    # 运行单个配准
    main()
    
    # 或运行批量配准
    # batch_registration_example()