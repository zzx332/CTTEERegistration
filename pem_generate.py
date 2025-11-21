"""
Structured Decision Forest for Probabilistic Edge Map Generation
用于CT和超声图像的边缘检测
"""

import numpy as np
import SimpleITK as sitk
from sklearn.ensemble import RandomForestClassifier
from scipy import ndimage
from typing import List, Tuple, Dict
import cv2
import os
import pickle
from pathlib import Path

class StructuredDecisionForest:
    """
    结构化决策森林用于概率边缘图生成
    
    特点：
    1. 输出空间是结构化的边缘patch而不是单个像素标签
    2. 每个叶节点存储边缘patch的分布
    3. 可以处理CT和超声的不同模态
    """
    
    def __init__(
        self,
        n_trees: int = 10,
        max_depth: int = 15,
        patch_size: int = 16,
        n_feature_samples: int = 2000,
        modality: str = 'CT'  # 'CT' or 'US'
    ):
        """
        Args:
            n_trees: 树的数量
            max_depth: 树的最大深度
            patch_size: 输出边缘patch的大小
            n_feature_samples: 每个节点测试的特征数量
            modality: 图像模态（CT或超声）
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.patch_size = patch_size
        self.n_feature_samples = n_feature_samples
        self.modality = modality
        
        # 存储训练好的树
        self.trees = []
        
        # 特征提取参数
        self.feature_channels = self._get_feature_channels()
    
    def _get_feature_channels(self) -> int:
        """
        根据模态确定特征通道数
        
        CT特征：
        - 原始强度
        - Sobel X/Y
        - 高斯平滑（多尺度）
        - Hessian特征
        
        超声特征：
        - 原始强度
        - 斑点噪声抑制
        - 方向梯度
        - 局部统计特征
        """
        if self.modality == 'CT':
            return 9  # 原始 + Sobel(2) + Gaussian(3) + Hessian(3)
        else:  # US
            return 10  # 原始 + 去噪 + 梯度(2) + 统计(6)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取多通道特征
        
        Args:
            image: 输入图像 (H, W)
        
        Returns:
            features: 特征图 (H, W, C)
        """
        features = []
        
        if self.modality == 'CT':
            # 1. 原始强度
            features.append(image)
            
            # 2. Sobel梯度
            sobel_x = ndimage.sobel(image, axis=1)
            sobel_y = ndimage.sobel(image, axis=0)
            features.extend([sobel_x, sobel_y])
            
            # 3. 多尺度高斯平滑
            for sigma in [1.0, 2.0, 4.0]:
                gaussian = ndimage.gaussian_filter(image, sigma=sigma)
                features.append(gaussian)
            
            # 4. Hessian特征（二阶导数）
            from scipy.ndimage import gaussian_laplace
            laplacian = gaussian_laplace(image, sigma=2.0)
            features.append(laplacian)
            
            # 梯度幅值
            gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            features.append(gradient_mag)
            
            # 梯度方向
            gradient_dir = np.arctan2(sobel_y, sobel_x)
            features.append(gradient_dir)
        
        else:  # 超声
            # 1. 原始强度
            features.append(image)
            
            # 2. 斑点噪声抑制（中值滤波）
            denoised = ndimage.median_filter(image, size=3)
            features.append(denoised)
            
            # 3. 梯度
            sobel_x = ndimage.sobel(denoised, axis=1)
            sobel_y = ndimage.sobel(denoised, axis=0)
            features.extend([sobel_x, sobel_y])
            
            # 4. 局部统计特征
            from scipy.ndimage import generic_filter
            
            # 局部均值
            local_mean = ndimage.uniform_filter(image, size=5)
            features.append(local_mean)
            
            # 局部标准差
            local_std = generic_filter(image, np.std, size=5)
            features.append(local_std)
            
            # 局部最大值
            local_max = ndimage.maximum_filter(image, size=5)
            features.append(local_max)
            
            # 局部最小值
            local_min = ndimage.minimum_filter(image, size=5)
            features.append(local_min)
            
            # 梯度幅值
            gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            features.append(gradient_mag)
            
            # 局部对比度
            local_contrast = local_max - local_min
            features.append(local_contrast)
        
        # 堆叠所有特征
        feature_map = np.stack(features, axis=-1)
        
        return feature_map.astype(np.float32)
    
    def sample_features_at_pixel(
        self,
        feature_map: np.ndarray,
        pixel_pos: Tuple[int, int],
        n_samples: int = 100
    ) -> np.ndarray:
        """
        在像素位置采样上下文特征
        
        使用偏移量采样周围像素的特征差异
        
        Args:
            feature_map: 特征图 (H, W, C)
            pixel_pos: 像素位置 (y, x)
            n_samples: 采样数量
        
        Returns:
            features: 特征向量 (n_samples,)
        """
        h, w, c = feature_map.shape
        y, x = pixel_pos
        
        features = []
        
        # 随机采样偏移量对
        max_offset = 15  # 最大偏移量
        
        for _ in range(n_samples):
            # 随机选择通道
            channel = np.random.randint(0, c)
            
            # 随机选择两个偏移量
            dy1 = np.random.randint(-max_offset, max_offset + 1)
            dx1 = np.random.randint(-max_offset, max_offset + 1)
            dy2 = np.random.randint(-max_offset, max_offset + 1)
            dx2 = np.random.randint(-max_offset, max_offset + 1)
            
            # 计算偏移后的坐标（带边界检查）
            y1 = np.clip(y + dy1, 0, h - 1)
            x1 = np.clip(x + dx1, 0, w - 1)
            y2 = np.clip(y + dy2, 0, h - 1)
            x2 = np.clip(x + dx2, 0, w - 1)
            
            # 特征差异
            feature_diff = feature_map[y1, x1, channel] - feature_map[y2, x2, channel]
            features.append(feature_diff)
        
        return np.array(features)
    
    def extract_edge_patch(
        self,
        edge_map: np.ndarray,
        center: Tuple[int, int]
    ) -> np.ndarray:
        """
        提取边缘patch作为结构化标签
        
        Args:
            edge_map: 边缘真值图 (H, W)
            center: patch中心 (y, x)
        
        Returns:
            patch: 边缘patch (patch_size, patch_size)
        """
        y, x = center
        half_size = self.patch_size // 2
        
        h, w = edge_map.shape
        
        # 提取patch（带边界处理）
        y_start = max(0, y - half_size)
        y_end = min(h, y + half_size)
        x_start = max(0, x - half_size)
        x_end = min(w, x + half_size)
        
        patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        
        # 计算在patch中的位置
        py_start = half_size - (y - y_start)
        py_end = py_start + (y_end - y_start)
        px_start = half_size - (x - x_start)
        px_end = px_start + (x_end - x_start)
        
        patch[py_start:py_end, px_start:px_end] = edge_map[y_start:y_end, x_start:x_end]
        
        return patch
    
    def train(
        self,
        images: List[np.ndarray],
        edge_maps: List[np.ndarray],
        n_samples_per_image: int = 500,
        downsample_factor: int = 2  # 新增参数
    ):
        """
        训练结构化决策森林（支持下采样加速）
        
        Args:
            images: 训练图像列表
            edge_maps: 对应的边缘真值图列表
            n_samples_per_image: 每张图像采样的像素数
            downsample_factor: 下采样倍数（1=不下采样，2=缩小2倍）
        """
        print(f"训练 {self.n_trees} 棵树...")
        if downsample_factor > 1:
            print(f"  使用 {downsample_factor}x 下采样加速训练")
        
        X_all = []
        Y_all = []
        
        for img_idx, (image, edge_map) in enumerate(zip(images, edge_maps)):
            print(f"  处理图像 {img_idx + 1}/{len(images)}...")
            
            # ========== 下采样 ==========
            if downsample_factor > 1:
                from scipy.ndimage import zoom
                
                print(f"    原始尺寸: {image.shape}")
                
                # 下采样图像（双线性插值）
                working_image = zoom(image, 1.0 / downsample_factor, order=1)
                
                # 下采样边缘标签（最近邻插值，保持二值性）
                working_edge_map = zoom(edge_map, 1.0 / downsample_factor, order=0)
                
                # 确保边缘标签仍然是二值的
                working_edge_map = (working_edge_map > 0.5).astype(np.float32)
                
                print(f"    下采样后: {working_image.shape}")
            else:
                working_image = image
                working_edge_map = edge_map
            # ============================
            
            # 提取特征
            feature_map = self.extract_features(working_image)
            h, w = working_image.shape
            
            # ========== 平衡采样 ==========
            half_size = self.patch_size // 2
            valid_y = np.arange(half_size, h - half_size)
            valid_x = np.arange(half_size, w - half_size)
            
            # 网格采样
            stride = max(1, half_size // 2)
            y_grid = valid_y[::stride]
            x_grid = valid_x[::stride]
            
            # 分类位置
            edge_positions = []
            non_edge_positions = []
            
            for y in y_grid:
                for x in x_grid:
                    patch = working_edge_map[y-half_size:y+half_size, x-half_size:x+half_size]
                    edge_ratio = np.sum(patch > 0) / patch.size
                    
                    if edge_ratio > 0.1:
                        edge_positions.append((y, x))
                    elif edge_ratio == 0:
                        non_edge_positions.append((y, x))
            
            print(f"    候选: {len(edge_positions)}边缘, {len(non_edge_positions)}非边缘")
            
            # 平衡采样
            n_edge = min(n_samples_per_image // 2, len(edge_positions))
            n_non_edge = min(n_samples_per_image // 2, len(non_edge_positions))
            
            if n_edge == 0 or n_non_edge == 0:
                print(f"    ⚠️ 跳过此图像: 样本不足")
                continue
            
            # 随机选择并立即打乱
            edge_indices = np.random.choice(len(edge_positions), n_edge, replace=False)
            non_edge_indices = np.random.choice(len(non_edge_positions), n_non_edge, replace=False)
            
            sample_positions = [edge_positions[i] for i in edge_indices] + \
                            [non_edge_positions[i] for i in non_edge_indices]
            
            np.random.shuffle(sample_positions)  # 立即打乱
            
            print(f"    ✓ 采样: {n_edge}边缘 + {n_non_edge}非边缘")
            # ==============================
            
            # 提取特征和标签
            for pos in sample_positions:
                # 特征
                features = self.sample_features_at_pixel(feature_map, pos, self.n_feature_samples)
                X_all.append(features)
                
                # 结构化标签（边缘patch）
                edge_patch = self.extract_edge_patch(working_edge_map, pos)
                Y_all.append(edge_patch.flatten())
        
        # 转换为numpy数组
        X_all = np.array(X_all)
        Y_all = np.array(Y_all)
        
        print(f"\n  训练数据: {X_all.shape}, 标签: {Y_all.shape}")
        
        # 打乱数据
        print("  打乱训练数据...")
        for _ in range(3):  # 多次打乱
            shuffle_indices = np.random.permutation(len(X_all))
            X_all = X_all[shuffle_indices]
            Y_all = Y_all[shuffle_indices]
        print("  ✓ 数据已充分打乱")
        
        # 转换为标签
        Y_labels = self._patches_to_labels(Y_all)
        unique, counts = np.unique(Y_labels, return_counts=True)
        print(f"  标签分布: {dict(zip(unique, counts))}")
        
        if len(unique) < 2:
            raise ValueError("训练数据只有一个类别！")
        
        # 训练每棵树
        self.trees = []
        for tree_idx in range(self.n_trees):
            print(f"  训练树 {tree_idx + 1}/{self.n_trees}...")
            
            tree = RandomForestClassifier(
                n_estimators=1,
                max_depth=self.max_depth,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=tree_idx
            )
            
            tree.fit(X_all, Y_labels)
            
            # ========== 关键：存储每个叶节点的patch分布 ==========
            leaf_ids = tree.apply(X_all)  # 获取每个样本的叶节点ID
            unique_leaves = np.unique(leaf_ids)
            
            leaf_patch_distribution = {}
            for leaf_id in unique_leaves:
                # 找到到达该叶节点的所有样本
                mask = (leaf_ids == leaf_id)
                # leaf_patches = Y_all[mask]  # 这些样本的patch
                leaf_patches = Y_all[mask[:,0]] 
                
                # 存储patch分布
                leaf_patch_distribution[leaf_id] = {
                    'mean_patch': np.mean(leaf_patches, axis=0),  # 平均patch
                    'patches': leaf_patches,  # 所有patch（可选，占内存）
                    'n_samples': len(leaf_patches)
                }
            
            self.trees.append((tree, leaf_patch_distribution))
            # =====================================================
        
        print("✓ 训练完成！")
    
    def _patches_to_labels(self, patches: np.ndarray) -> np.ndarray:
        """将patch转换为离散标签（简化）"""
        # 简化：使用patch的主成分或聚类
        # 实际SDF会保留完整分布
        # return (patches.sum(axis=1) > patches.shape[1] * 0.3).astype(int)
        return (patches.sum(axis=1) > 0).astype(int)
    
    def predict_edge_probability(
        self, 
        image: np.ndarray,
        downsample_factor: int = 2  # 下采样倍数
    ) -> np.ndarray:
        """
        预测概率边缘图（支持下采样加速）
        
        Args:
            image: 输入图像 (H, W)
            downsample_factor: 下采样倍数（2=缩小2倍，速度提升4倍）
        
        Returns:
            edge_prob_map: 概率边缘图 (H, W)，原始分辨率
        """
        print("生成概率边缘图...")
        
        original_shape = image.shape
        
        # 1. 下采样图像
        if downsample_factor > 1:
            print(f"  下采样 {downsample_factor}x: {image.shape} -> ", end="")
            
            # 方法A：使用scipy进行平滑下采样
            from scipy.ndimage import zoom
            downsampled_image = zoom(
                image, 
                1.0 / downsample_factor, 
                order=1  # 双线性插值
            )
            
            # 或方法B：使用SimpleITK（更适合医学图像）
            # downsampled_image = self._downsample_image(image, downsample_factor)
            
            print(f"{downsampled_image.shape}")
            working_image = downsampled_image
        else:
            working_image = image
        
        # 2. 提取特征（在下采样后的图像上）
        feature_map = self.extract_features(working_image)
        h, w = working_image.shape

        # 3. 初始化概率图
        edge_prob_map = np.zeros((h, w), dtype=np.float32)
        vote_count = np.zeros((h, w), dtype=np.float32)
        
        half_size = self.patch_size // 2
        stride = self.patch_size // 4
        total_patches = (h - half_size * 2) // stride * (w - half_size * 2) // stride
        current_patch = 0
        
        print(f"  处理 {total_patches} 个patches (stride={stride})...")
        for y in range(half_size, h - half_size, stride):
            for x in range(half_size, w - half_size, stride):
                features = self.sample_features_at_pixel(
                    feature_map, (y, x), self.n_feature_samples
                ).reshape(1, -1)
                
                # 所有树投票
                patch_predictions = []
                for tree, leaf_patch_distribution in self.trees:
                    # ========== 关键：获取叶节点的patch分布 ==========
                    leaf_id = tree.apply(features)[0]
                    if isinstance(leaf_id, np.ndarray):
                        leaf_id = int(leaf_id.item())  # 或 leaf_id = leaf_id.item()
                    else:
                        leaf_id = int(leaf_id)
                    # 获取该叶节点的平均patch
                    if leaf_id in leaf_patch_distribution:
                        mean_patch = leaf_patch_distribution[leaf_id]['mean_patch']
                        # 重塑为2D patch
                        patch_2d = mean_patch.reshape(self.patch_size, self.patch_size)
                        patch_predictions.append(patch_2d)
                    else:
                        # 叶节点没有数据，使用默认值
                        patch_predictions.append(np.zeros((self.patch_size, self.patch_size)))
                    # ================================================
                
                # 平均所有树的patch预测
                avg_patch = np.mean(patch_predictions, axis=0)
                
                # 更新概率图（叠加完整的patch）
                y_start = y - half_size
                y_end = y + half_size
                x_start = x - half_size
                x_end = x + half_size
                
                edge_prob_map[y_start:y_end, x_start:x_end] += avg_patch
                vote_count[y_start:y_end, x_start:x_end] += 1        
                current_patch += 1
                print(f"patch_votes finished! {current_patch}/{total_patches}")
        # 5. 归一化
        edge_prob_map = np.divide(
            edge_prob_map,
            vote_count,
            out=np.zeros_like(edge_prob_map),
            where=vote_count > 0
        )
        
        # 6. 上采样回原始分辨率
        if downsample_factor > 1:
            print(f"  上采样回原始分辨率: {edge_prob_map.shape} -> {original_shape}")
            
            from scipy.ndimage import zoom
            edge_prob_map = zoom(
                edge_prob_map,
                downsample_factor,
                order=1  # 双线性插值
            )
            
            # 确保尺寸完全匹配（处理可能的舍入误差）
            if edge_prob_map.shape != original_shape:
                from skimage.transform import resize
                edge_prob_map = resize(
                    edge_prob_map,
                    original_shape,
                    order=1,
                    preserve_range=True,
                    anti_aliasing=False
                )
        
        print("✓ 概率边缘图生成完成！")
        
        return edge_prob_map


    def _downsample_image(self, image: np.ndarray, factor: int) -> np.ndarray:
        """
        使用SimpleITK下采样（可选方法）
        """
        import SimpleITK as sitk
        
        # 转换为SimpleITK图像
        sitk_img = sitk.GetImageFromArray(image)
        
        # 设置新的spacing
        original_spacing = sitk_img.GetSpacing()
        new_spacing = [s * factor for s in original_spacing]
        
        # 计算新的尺寸
        original_size = sitk_img.GetSize()
        new_size = [int(s / factor) for s in original_size]
        
        # 重采样
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        downsampled = resampler.Execute(sitk_img)
        
        return sitk.GetArrayFromImage(downsampled)

    def save_model(self, filepath: str):
        """
        保存训练好的模型
        
        Args:
            filepath: 保存路径，如 "sdf_ct_model.pkl"
        """
        model_data = {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'patch_size': self.patch_size,
            'n_feature_samples': self.n_feature_samples,
            'modality': self.modality,
            'trees': self.trees,
            'feature_channels': self.feature_channels
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ 模型已保存到: {filepath}")
        print(f"  文件大小: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        加载训练好的模型
        
        Args:
            filepath: 模型文件路径
        
        Returns:
            加载的SDF模型
        """
        print(f"加载模型: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 创建实例
        sdf = cls(
            n_trees=model_data['n_trees'],
            max_depth=model_data['max_depth'],
            patch_size=model_data['patch_size'],
            n_feature_samples=model_data['n_feature_samples'],
            modality=model_data['modality']
        )
        
        # 恢复训练好的树
        sdf.trees = model_data['trees']
        sdf.feature_channels = model_data['feature_channels']
        
        print(f"✓ 模型加载完成")
        print(f"  树的数量: {len(sdf.trees)}")
        print(f"  模态: {sdf.modality}")
        
        return sdf

# ============================================================================
# 使用示例
# ============================================================================

def create_edge_ground_truth(image: np.ndarray, method='canny') -> np.ndarray:
    """
    创建边缘真值（用于训练）
    
    可以使用：
    1. Canny边缘检测
    2. 手动标注
    3. 从分割mask生成
    """
    if method == 'canny':
        # 归一化到0-255
        img_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        edges = cv2.Canny(img_norm, 50, 150)
        return (edges > 0).astype(np.float32)
    
    elif method == 'sobel':
        sobel_x = ndimage.sobel(image, axis=1)
        sobel_y = ndimage.sobel(image, axis=0)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        threshold = np.percentile(gradient_mag, 90)
        return (gradient_mag > threshold).astype(np.float32)


def main_ct_edge_detection():
    """CT边缘检测示例"""
    
    # 1. 加载训练数据
    print("=" * 60)
    print("CT概率边缘图生成")
    print("=" * 60)
    
    # 加载CT图像
    ct_images = []
    edge_maps = []
    ct_dir = r"D:\dataset\TEECT_data\ct\ct_train_1004_image"
    ct_paths = os.listdir(ct_dir)
    # 示例：加载多张CT切片
    ct_paths = [
        os.path.join(ct_dir, path) for path in ct_paths
    ][::4]
    # ct_paths = [ct_paths]
    for path in ct_paths:
        img = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(img)  # 2D切片
        ct_images.append(img_array)
        
        # 生成边缘真值
        # edge_map = create_edge_ground_truth(img_array, method='canny')
        edge_path = path.replace('_image', '_edge').replace('.nii.gz', '_edge.nii.gz')
        img_edge = sitk.ReadImage(edge_path)
        img_edge_array = sitk.GetArrayFromImage(img_edge)  # 2D切片
        edge_maps.append(img_edge_array)
    
    # 2. 训练SDF
    sdf_ct = StructuredDecisionForest(
        n_trees=10,
        max_depth=15,
        patch_size=16,
        modality='CT'
    )
    ct_model_path = r"D:\dataset\TEECT_data\pem\ct_sdf_model.pkl"
    # sdf_ct = StructuredDecisionForest.load_model(ct_model_path)
    sdf_ct.train(ct_images, edge_maps, n_samples_per_image=100, downsample_factor=2)
    sdf_ct.save_model(ct_model_path)
    # 3. 预测新图像
    test_img = sitk.ReadImage(r"D:\dataset\TEECT_data\ct\ct_train_1002_image\slice_062_t0.0_rx0_ry0.nii.gz")
    test_array = sitk.GetArrayFromImage(test_img)
    
    edge_prob = sdf_ct.predict_edge_probability(test_array)
    
    # 4. 保存结果
    edge_prob_img = sitk.GetImageFromArray(edge_prob)
    edge_prob_img.CopyInformation(test_img)
    sitk.WriteImage(edge_prob_img, r"D:\dataset\TEECT_data\pem\ct_edge_probability.nii.gz")
    
    print("✓ CT边缘概率图已保存")


def main_us_edge_detection():
    """超声边缘检测示例"""
    
    print("=" * 60)
    print("超声概率边缘图生成")
    print("=" * 60)
    
    # 类似CT的流程，但使用超声特定的特征
    sdf_us = StructuredDecisionForest(
        n_trees=10,
        max_depth=15,
        patch_size=16,
        modality='US'
    )
    
    # ... 训练和预测 ...


if __name__ == "__main__":
    # 运行示例
    main_ct_edge_detection()
    # main_us_edge_detection()