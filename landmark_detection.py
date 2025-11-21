"""
基于结构化回归的解剖关键点检测
用于心脏CT图像的心尖点、瓣环点等关键点定位
"""

import numpy as np
import SimpleITK as sitk
from sklearn.ensemble import RandomForestRegressor
from scipy import ndimage
from typing import List, Tuple, Dict
import cv2
from pathlib import Path
import pickle
from collections import defaultdict


class StructuredLandmarkDetector:
    """
    基于结构化回归森林的关键点检测器
    
    特点：
    1. 输出空间是位移向量而不是类别标签
    2. 每个叶节点存储位移向量分布
    3. 使用投票聚合机制定位关键点
    """
    
    def __init__(
        self,
        n_trees: int = 20,
        max_depth: int = 20,
        n_feature_samples: int = 2000,
        modality: str = 'CT'
    ):
        """
        Args:
            n_trees: 树的数量
            max_depth: 树的最大深度
            n_feature_samples: 每个节点测试的特征数量
            modality: 图像模态
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_feature_samples = n_feature_samples
        self.modality = modality
        
        # 存储训练好的树 (每个landmark一组树)
        self.landmark_forests = {}  # {landmark_name: [(tree, leaf_displacements), ...]}
        
        # 定义关键点
        self.landmark_names = [
            'apex',          # 心尖点
            'mitral_anterior',   # 二尖瓣前叶
            'mitral_posterior',  # 二尖瓣后叶
            'tricuspid_anterior',  # 三尖瓣前叶
            'tricuspid_septal',    # 三尖瓣间隔叶
            'aortic_center'     # 主动脉瓣中心
        ]
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取多通道特征（与SDF类似）
        
        Returns:
            features: 特征图 (H, W, C)
        """
        features = []
        
        # 1. 原始强度
        features.append(image)
        
        # 2. Sobel梯度
        sobel_x = ndimage.sobel(image, axis=1)
        sobel_y = ndimage.sobel(image, axis=0)
        features.extend([sobel_x, sobel_y])
        
        # 3. 多尺度高斯平滑
        for sigma in [1.0, 2.0, 4.0, 8.0]:
            gaussian = ndimage.gaussian_filter(image, sigma=sigma)
            features.append(gaussian)
        
        # 4. Hessian特征
        from scipy.ndimage import gaussian_laplace
        laplacian = gaussian_laplace(image, sigma=2.0)
        features.append(laplacian)
        
        # 5. 梯度幅值和方向
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_dir = np.arctan2(sobel_y, sobel_x)
        features.extend([gradient_mag, gradient_dir])
        
        # 6. 局部统计特征
        local_mean = ndimage.uniform_filter(image, size=5)
        local_std = ndimage.generic_filter(image, np.std, size=5)
        features.extend([local_mean, local_std])
        
        # 堆叠所有特征
        feature_map = np.stack(features, axis=-1)
        
        return feature_map.astype(np.float32)
    
    def sample_features_at_pixel(
        self,
        feature_map: np.ndarray,
        pixel_pos: Tuple[int, int],
        n_samples: int = 100
    ) -> np.ndarray:
        """在像素位置采样上下文特征（特征差异）"""
        h, w, c = feature_map.shape
        y, x = pixel_pos
        
        features = []
        max_offset = 30  # 增大偏移量以捕获更大的上下文
        
        for _ in range(n_samples):
            channel = np.random.randint(0, c)
            
            dy1 = np.random.randint(-max_offset, max_offset + 1)
            dx1 = np.random.randint(-max_offset, max_offset + 1)
            dy2 = np.random.randint(-max_offset, max_offset + 1)
            dx2 = np.random.randint(-max_offset, max_offset + 1)
            
            y1 = np.clip(y + dy1, 0, h - 1)
            x1 = np.clip(x + dx1, 0, w - 1)
            y2 = np.clip(y + dy2, 0, h - 1)
            x2 = np.clip(x + dx2, 0, w - 1)
            
            feature_diff = feature_map[y1, x1, channel] - feature_map[y2, x2, channel]
            features.append(feature_diff)
        
        return np.array(features)
    
    def compute_displacement_vector(
        self,
        sample_pos: Tuple[int, int],
        landmark_pos: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        计算从采样位置到关键点的位移向量
        
        Args:
            sample_pos: 采样位置 (y, x)
            landmark_pos: 关键点位置 (y, x)
        
        Returns:
            displacement: (Δx, Δy)
        """
        dy = landmark_pos[0] - sample_pos[0]
        dx = landmark_pos[1] - sample_pos[1]
        return (dx, dy)
    
    def train(
        self,
        images: List[np.ndarray],
        landmarks: List[Dict[str, Tuple[int, int]]],
        n_samples_per_image: int = 1000,
        sampling_radius: int = 50
    ):
        """
        训练关键点检测器
        
        Args:
            images: 训练图像列表
            landmarks: 关键点标注列表，每个元素是字典 {'apex': (y, x), 'mitral_anterior': (y, x), ...}
            n_samples_per_image: 每张图像采样的像素数
            sampling_radius: 关键点周围的采样半径（像素）
        """
        print(f"训练关键点检测器...")
        print(f"  图像数量: {len(images)}")
        print(f"  关键点类型: {list(landmarks[0].keys())}")
        
        # 为每个关键点收集训练数据
        landmark_data = {name: {'X': [], 'Y': []} for name in self.landmark_names}
        
        for img_idx, (image, lm_dict) in enumerate(zip(images, landmarks)):
            print(f"\n  处理图像 {img_idx + 1}/{len(images)}...")
            
            # 提取特征
            feature_map = self.extract_features(image)
            h, w = image.shape
            
            # 为每个关键点采样
            for lm_name, lm_pos in lm_dict.items():
                if lm_name not in self.landmark_names:
                    continue
                
                if lm_pos is None:  # 有些图像可能没有某些关键点
                    continue
                
                lm_y, lm_x = lm_pos
                
                # 在关键点周围采样
                sample_positions = []
                
                # 方法1：在关键点周围的圆形区域内采样
                for _ in range(n_samples_per_image):
                    # 随机半径和角度
                    r = np.random.uniform(0, sampling_radius)
                    theta = np.random.uniform(0, 2 * np.pi)
                    
                    dy = int(r * np.sin(theta))
                    dx = int(r * np.cos(theta))
                    
                    sample_y = np.clip(lm_y + dy, 0, h - 1)
                    sample_x = np.clip(lm_x + dx, 0, w - 1)
                    
                    sample_positions.append((sample_y, sample_x))
                
                # 提取特征和位移向量
                for sample_pos in sample_positions:
                    # 特征
                    features = self.sample_features_at_pixel(
                        feature_map, sample_pos, self.n_feature_samples
                    )
                    
                    # 位移向量
                    displacement = self.compute_displacement_vector(sample_pos, lm_pos)
                    
                    landmark_data[lm_name]['X'].append(features)
                    landmark_data[lm_name]['Y'].append(displacement)
                
                print(f"    {lm_name}: 采样 {len(sample_positions)} 个点")
        
        # 训练每个关键点的森林
        for lm_name in self.landmark_names:
            if len(landmark_data[lm_name]['X']) == 0:
                print(f"\n  跳过 {lm_name}: 无训练数据")
                continue
            
            print(f"\n  训练 {lm_name} 的森林...")
            
            X = np.array(landmark_data[lm_name]['X'])
            Y = np.array(landmark_data[lm_name]['Y'])  # (N, 2): (Δx, Δy)
            
            print(f"    训练数据: {X.shape}, 位移: {Y.shape}")
            
            # 训练多棵树
            trees = []
            for tree_idx in range(self.n_trees):
                print(f"    训练树 {tree_idx + 1}/{self.n_trees}...")
                
                # 使用RandomForestRegressor的单棵树
                tree = RandomForestRegressor(
                    n_estimators=1,
                    max_depth=self.max_depth,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=tree_idx
                )
                
                tree.fit(X, Y)
                
                # 存储每个叶节点的位移向量分布
                leaf_ids = tree.apply(X)
                unique_leaves = np.unique(leaf_ids)
                
                leaf_displacements = {}
                for leaf_id in unique_leaves:
                    mask = (leaf_ids == leaf_id)
                    if isinstance(mask, np.ndarray) and mask.ndim > 1:
                        mask = mask[:, 0]
                    
                    leaf_samples = Y[mask]
                    
                    # 存储位移向量的均值和方差
                    leaf_displacements[int(leaf_id)] = {
                        'mean': np.mean(leaf_samples, axis=0),
                        'std': np.std(leaf_samples, axis=0),
                        'samples': leaf_samples,  # 保留所有样本用于投票
                        'n_samples': len(leaf_samples)
                    }
                
                trees.append((tree, leaf_displacements))
            
            self.landmark_forests[lm_name] = trees
        
        print("\n✓ 训练完成！")
    
    def predict_landmark(
        self,
        image: np.ndarray,
        landmark_name: str,
        search_grid_stride: int = 5,
        voting_threshold: float = 0.1
    ) -> Tuple[int, int, float]:
        """
        预测单个关键点位置
        
        Args:
            image: 输入图像
            landmark_name: 关键点名称
            search_grid_stride: 搜索网格步长
            voting_threshold: 投票阈值（归一化）
        
        Returns:
            (y, x, confidence): 关键点位置和置信度
        """
        if landmark_name not in self.landmark_forests:
            raise ValueError(f"未训练的关键点: {landmark_name}")
        
        print(f"预测 {landmark_name}...")
        
        # 提取特征
        feature_map = self.extract_features(image)
        h, w = image.shape
        
        # 初始化投票图（Hough voting space）
        vote_map = np.zeros((h, w), dtype=np.float32)
        
        # 在网格上采样
        y_coords = range(0, h, search_grid_stride)
        x_coords = range(0, w, search_grid_stride)
        total_samples = len(y_coords) * len(x_coords)
        
        print(f"  在 {total_samples} 个位置进行投票...")
        
        sample_count = 0
        for y in y_coords:
            for x in x_coords:
                # 提取特征
                features = self.sample_features_at_pixel(
                    feature_map, (y, x), self.n_feature_samples
                ).reshape(1, -1)
                
                # 所有树投票
                for tree, leaf_displacements in self.landmark_forests[landmark_name]:
                    # 找到叶节点
                    leaf_id = tree.apply(features)[0]
                    if isinstance(leaf_id, np.ndarray):
                        leaf_id = int(leaf_id.item())
                    else:
                        leaf_id = int(leaf_id)
                    
                    # 获取位移向量
                    if leaf_id in leaf_displacements:
                        mean_displacement = leaf_displacements[leaf_id]['mean']
                        
                        # 计算预测位置
                        pred_x = x + mean_displacement[0]
                        pred_y = y + mean_displacement[1]
                        
                        # 投票（带高斯权重）
                        pred_x_int = int(round(pred_x))
                        pred_y_int = int(round(pred_y))
                        
                        if 0 <= pred_y_int < h and 0 <= pred_x_int < w:
                            # 使用不确定性加权
                            n_samples = leaf_displacements[leaf_id]['n_samples']
                            weight = np.sqrt(n_samples)  # 样本越多权重越大
                            
                            vote_map[pred_y_int, pred_x_int] += weight
                
                sample_count += 1
                if sample_count % 100 == 0:
                    print(f"    进度: {sample_count}/{total_samples}")
        
        # 归一化投票图
        if vote_map.max() > 0:
            vote_map = vote_map / vote_map.max()
        
        # 找到投票最高的位置
        max_vote_pos = np.unravel_index(np.argmax(vote_map), vote_map.shape)
        max_vote_value = vote_map[max_vote_pos]
        
        print(f"  ✓ 检测到 {landmark_name} 在 ({max_vote_pos[0]}, {max_vote_pos[1]}), 置信度={max_vote_value:.3f}")
        
        return (max_vote_pos[0], max_vote_pos[1], max_vote_value)
    
    def predict_all_landmarks(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Dict[str, Tuple[int, int, float]]:
        """预测所有关键点"""
        results = {}
        
        for lm_name in self.landmark_names:
            if lm_name in self.landmark_forests:
                try:
                    results[lm_name] = self.predict_landmark(image, lm_name, **kwargs)
                except Exception as e:
                    print(f"  ✗ {lm_name} 检测失败: {e}")
                    results[lm_name] = None
        
        return results
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'n_feature_samples': self.n_feature_samples,
            'modality': self.modality,
            'landmark_names': self.landmark_names,
            'landmark_forests': self.landmark_forests
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ 模型已保存到: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        print(f"加载模型: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        detector = cls(
            n_trees=model_data['n_trees'],
            max_depth=model_data['max_depth'],
            n_feature_samples=model_data['n_feature_samples'],
            modality=model_data['modality']
        )
        
        detector.landmark_names = model_data['landmark_names']
        detector.landmark_forests = model_data['landmark_forests']
        
        print(f"✓ 模型加载完成，包含 {len(detector.landmark_forests)} 个关键点")
        
        return detector


# ============================================================================
# 使用示例
# ============================================================================

def extract_landmarks_from_segmentation(seg_array: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """
    从分割标签中提取关键点位置
    
    Args:
        seg_array: 分割标签 (H, W)，包含不同腔室的标签
    
    Returns:
        landmarks: 关键点字典
    """
    landmarks = {}
    
    # 假设标签定义：
    # 420: 左心房(LA), 500: 左心室(LV), 550: 右心房(RA), 600: 右心室(RV)
    
    # 1. 心尖点：左心室最下方的点
    lv_mask = (seg_array == 500)
    if np.any(lv_mask):
        lv_coords = np.argwhere(lv_mask)
        apex_idx = np.argmax(lv_coords[:, 0])  # 最大row = 最下方
        landmarks['apex'] = tuple(lv_coords[apex_idx])
    
    # 2. 二尖瓣环：左心房和左心室交界处
    la_mask = (seg_array == 420)
    if np.any(lv_mask) and np.any(la_mask):
        # 膨胀LV和LA，找交界
        from scipy.ndimage import binary_dilation
        lv_dilated = binary_dilation(lv_mask)
        la_dilated = binary_dilation(la_mask)
        mitral_boundary = lv_dilated & la_dilated & ~lv_mask & ~la_mask
        
        if np.any(mitral_boundary):
            mitral_coords = np.argwhere(mitral_boundary)
            # 前叶：最左侧
            mitral_anterior_idx = np.argmin(mitral_coords[:, 1])
            landmarks['mitral_anterior'] = tuple(mitral_coords[mitral_anterior_idx])
            # 后叶：最右侧
            mitral_posterior_idx = np.argmax(mitral_coords[:, 1])
            landmarks['mitral_posterior'] = tuple(mitral_coords[mitral_posterior_idx])
    
    # 类似方法提取其他关键点...
    
    return landmarks


def main():
    """主函数示例"""
    
    # 1. 准备训练数据
    print("=" * 60)
    print("关键点检测训练")
    print("=" * 60)
    
    images = []
    landmarks_list = []
    
    # 加载数据
    data_dir = Path(r"D:\dataset\TEECT_data\ct\ct_train_1004_image")
    label_dir = Path(r"D:\dataset\TEECT_data\ct\ct_train_1004_label")
    
    image_files = sorted(data_dir.glob("*.nii*"))[:20]  # 使用前20张
    model_path = r"D:\dataset\TEECT_data\models\landmark_detector.pkl"
    if Path(model_path).exists():
        detector = StructuredLandmarkDetector.load_model(model_path)
    else:
        for img_file in image_files:
            # 加载图像
            img = sitk.ReadImage(str(img_file))
            img_array = sitk.GetArrayFromImage(img).squeeze()
            images.append(img_array)
            
            # 加载对应的分割标签并提取关键点
            label_file = label_dir / img_file.name
            if label_file.exists():
                label_img = sitk.ReadImage(str(label_file))
                label_array = sitk.GetArrayFromImage(label_img).squeeze()
                
                landmarks = extract_landmarks_from_segmentation(label_array)
                landmarks_list.append(landmarks)
            else:
                landmarks_list.append({})
        detector = StructuredLandmarkDetector(
            n_trees=20,
            max_depth=20,
            modality='CT'
        )
        detector.train(images, landmarks_list, n_samples_per_image=1000, sampling_radius=50)
        detector.save_model(model_path)

    
    # 4. 预测新图像
    test_img = sitk.ReadImage(r"D:\dataset\TEECT_data\ct\ct_train_1002_image\slice_062_t0.0_rx0_ry0.nii.gz")
    test_array = sitk.GetArrayFromImage(test_img).squeeze()
    
    results = detector.predict_all_landmarks(test_array, search_grid_stride=10)
    
    print("\n预测结果:")
    for lm_name, (y, x, conf) in results.items():
        if (y, x, conf) is not None:
            print(f"  {lm_name}: ({y}, {x}), 置信度={conf:.3f}")


if __name__ == "__main__":
    main()
