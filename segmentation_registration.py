import SimpleITK as sitk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import os 
from datetime import datetime
import pandas as pd


class SegmentationBasedRegistration2D:
    """
    基于多标签分割的2D图像刚性配准
    
    功能：
    - 支持多标签分割图配准（四腔心）
    - 使用Dice系数/Label Overlap作为相似性度量
    - 刚性变换（平移 + 旋转 + 可选缩放）
    - 直接优化心腔重叠度
    """
    
    def __init__(self, label_weights: Dict[int, float] = None):
        """
        初始化配准器
        
        Args:
            label_weights: 各标签的权重，如 {420: 1.0, 500: 1.5, 550: 1.0, 600: 1.0}
                          左心室(LV)可以设置更高权重
        """
        self.registration_method = None
        self.final_transform = None
        self.metric_values = []
        
        # 默认四腔心标签权重
        if label_weights is None:
            self.label_weights = {
                2: 1.0,  # 左心房 (LA)
                1: 1.5,  # 左心室 (LV) - 更重要
                3: 1.0,  # 右心房 (RA)
                4: 1.0   # 右心室 (RV)
            }
        else:
            self.label_weights = label_weights
    
    def compute_dice_coefficient(
        self,
        fixed_seg: sitk.Image,
        moving_seg: sitk.Image,
        label: int
    ) -> float:
        """
        计算单个标签的Dice系数
        
        Args:
            fixed_seg: 固定分割图
            moving_seg: 移动分割图
            label: 标签值
        
        Returns:
            Dice系数 [0, 1]
        """
        # 提取特定标签的二值mask
        fixed_array = sitk.GetArrayFromImage(fixed_seg)
        moving_array = sitk.GetArrayFromImage(moving_seg)
        
        fixed_mask = (fixed_array == label).astype(np.uint8)
        moving_mask = (moving_array == label).astype(np.uint8)
        
        # 计算Dice
        intersection = np.sum(fixed_mask * moving_mask)
        union = np.sum(fixed_mask) + np.sum(moving_mask)
        
        if union == 0:
            return 0.0
        
        dice = 2.0 * intersection / union
        return dice
    
    def compute_weighted_dice(
        self,
        fixed_seg: sitk.Image,
        moving_seg: sitk.Image
    ) -> float:
        """
        计算所有标签的加权平均Dice系数
        
        Returns:
            加权Dice系数
        """
        total_dice = 0.0
        total_weight = 0.0
        
        for label, weight in self.label_weights.items():
            dice = self.compute_dice_coefficient(fixed_seg, moving_seg, label)
            total_dice += dice * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_dice / total_weight
    
    def register(
        self,
        fixed_seg_path: str,
        moving_seg_path: str,
        output_dir: str = None,
        initial_transform: Optional[sitk.Transform] = None,
        number_of_iterations: int = 500,
        use_scaling: bool = False,
        save_visualization: bool = True
    ) -> Tuple[sitk.Image, sitk.Transform]:
        """
        执行基于分割的2D刚性配准
        
        Args:
            fixed_seg_path: 固定分割图路径（参考，通常是CT）
            moving_seg_path: 移动分割图路径（待配准，通常是TEE）
            output_dir: 输出目录
            initial_transform: 初始变换（可选）
            number_of_iterations: 最大迭代次数
            use_scaling: 是否使用缩放变换
            save_visualization: 是否保存可视化结果
            
        Returns:
            (配准后的分割图, 最终变换)
        """
        print("=" * 60)
        print("2D刚性配准 - 基于多标签分割 (Dice优化)")
        print("=" * 60)
        
        # 1. 读取分割图
        print("\n1. 读取分割图...")
        fixed_seg = sitk.ReadImage(fixed_seg_path, sitk.sitkUInt8)
        moving_seg = sitk.ReadImage(moving_seg_path, sitk.sitkUInt8)
        
        print(f"  固定分割: {fixed_seg.GetSize()}")
        print(f"  移动分割: {moving_seg.GetSize()}")
        
        # 坐标系转换（如果需要）
        print("  转换坐标系...")
        fixed_seg = sitk.Flip(fixed_seg, flipAxes=[True, True])
        fixed_seg.SetOrigin((0, 0))
        
        # 检查标签
        fixed_array = sitk.GetArrayFromImage(fixed_seg)
        moving_array = sitk.GetArrayFromImage(moving_seg)
        
        fixed_labels = np.unique(fixed_array[fixed_array > 0])
        moving_labels = np.unique(moving_array[moving_array > 0])
        
        print(f"  固定图像标签: {fixed_labels}")
        print(f"  移动图像标签: {moving_labels}")
        print(f"  标签权重: {self.label_weights}")
        
        # 2. 初始化配准方法
        print("\n2. 初始化配准方法...")
        registration = sitk.ImageRegistrationMethod()
        
        # ===== 关键：使用LabelOverlapMetric（Dice系数）=====
        # 注意：SimpleITK没有直接的Dice metric，我们使用自定义或LabelOverlap
        # 方法1：使用相关性（因为分割图标签固定，相关性也能工作）
        # 方法2：转换为距离图后使用MSE
        # 方法3：自定义Dice metric（推荐）
        
        # 这里我们使用方法2：距离场配准
        print("  生成距离场...")
        # fixed_distance = self._compute_multi_label_distance_map(fixed_seg)
        # moving_distance = self._compute_multi_label_distance_map(moving_seg)
        
        # 使用MSE度量（距离场越接近，分割越重叠）
        # registration.SetMetricAsMeanSquares()
        moving_seg = self._normalize_image_ncc(moving_seg)
        fixed_seg = self._normalize_image_ncc(fixed_seg)
        registration.SetMetricAsCorrelation()
        registration.SetMetricSamplingStrategy(registration.REGULAR)
        registration.SetMetricSamplingPercentage(0.5)
        
        # 或使用相关性
        # registration.SetMetricAsCorrelation()
        
        # 设置插值器（最近邻，保持标签）
        registration.SetInterpolator(sitk.sitkNearestNeighbor)
        
        # 设置优化器
        print("  设置优化器: Amoeba (Nelder-Mead)")
        registration.SetOptimizerAsAmoeba(
            simplexDelta=5.0,
            numberOfIterations=number_of_iterations,
            parametersConvergenceTolerance=0.01,
            functionConvergenceTolerance=0.001,
            withRestarts=False
        )
        
        # 3. 初始化变换
        if initial_transform is None:
            print("\n3. 初始变换（质心对齐）:")
            fixed_center = fixed_seg.GetOrigin() + (self._compute_centroid(fixed_seg)) * fixed_seg.GetSpacing()
            moving_center = moving_seg.GetOrigin() + (self._compute_centroid(moving_seg)) * moving_seg.GetSpacing()
            
            translation = (moving_center[0] - fixed_center[0], 
                          moving_center[1] - fixed_center[1])
            
            if use_scaling:
                initial_transform = sitk.Similarity2DTransform()
                initial_transform.SetScale(1.0)
            else:
                initial_transform = sitk.Euler2DTransform()
            
            initial_transform.SetCenter(moving_center)
            initial_transform.SetTranslation(translation)
            initial_transform.SetAngle(0)
            
            print(f"  固定质心: ({fixed_center[0]:.2f}, {fixed_center[1]:.2f})")
            print(f"  移动质心: ({moving_center[0]:.2f}, {moving_center[1]:.2f})")
            print(f"  初始平移: ({translation[0]:.2f}, {translation[1]:.2f})")
            
            # 计算初始Dice
            initial_resampled = sitk.Resample(
                moving_seg, fixed_seg, initial_transform,
                sitk.sitkNearestNeighbor, 0, moving_seg.GetPixelID()
            )
            initial_dice = self.compute_weighted_dice(fixed_seg, initial_resampled)
            print(f"  初始Dice: {initial_dice:.4f}")
        
        registration.SetInitialTransform(initial_transform, inPlace=False)
        
        # 设置优化器尺度
        if use_scaling:
            registration.SetOptimizerScales([5, 0.95, 0.1, 0.1])
        else:
            registration.SetOptimizerScales([10, 1.0, 1.0])
        
        # 多分辨率策略
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # 添加观察器
        self.metric_values = []
        self.dice_values = [] 



        
        
        def iteration_callback():
            metric = registration.GetMetricValue()
            self.metric_values.append(metric)
            
            # 每10次迭代计算一次Dice
            if len(self.metric_values) % 10 == 0:
                current_transform = registration.GetInitialTransform()
                temp_resampled = sitk.Resample(
                    moving_seg, fixed_seg, current_transform,
                    sitk.sitkNearestNeighbor, 0, moving_seg.GetPixelID()
                )
                dice = self.compute_weighted_dice(fixed_seg, temp_resampled)
                self.dice_values.append(dice)
                print(f"  迭代 {len(self.metric_values)}: Dice = {dice:.4f}, Metric = {metric:.6f}")
        
        registration.AddCommand(sitk.sitkIterationEvent, iteration_callback)
        
        # 4. 执行配准（使用距离场）
        print("\n4. 开始配准...")
        try:
            # final_transform = registration.Execute(fixed_distance, moving_distance)
            final_transform = registration.Execute(fixed_seg, moving_seg)
            
            print("\n5. 配准完成!")
            print(f"  优化器停止条件: {registration.GetOptimizerStopConditionDescription()}")
            print(f"  最终度量值: {registration.GetMetricValue():.6f}")
            
        except Exception as e:
            print(f"\n配准失败: {e}")
            raise
        
        # 5. 应用变换到分割图
        print("\n6. 应用变换...")
        resampled_seg = sitk.Resample(
            moving_seg,
            fixed_seg,
            final_transform,
            sitk.sitkNearestNeighbor,  # 保持标签
            0,
            moving_seg.GetPixelID()
        )
        
        # 计算最终Dice
        final_dice_per_label = {}
        for label in self.label_weights.keys():
            dice = self.compute_dice_coefficient(fixed_seg, resampled_seg, label)
            final_dice_per_label[label] = dice
            print(f"  标签 {label} Dice: {dice:.4f}")
        
        final_dice = self.compute_weighted_dice(fixed_seg, resampled_seg)
        print(f"  加权平均Dice: {final_dice:.4f}")
        
        # 打印变换参数
        print("\n7. 变换参数:")
        self._print_transform_parameters(initial_transform, "初始")
        self._print_transform_parameters(final_transform, "最终")
        
        # 保存结果
        if output_dir:
            self._save_results(
                fixed_seg, moving_seg, resampled_seg, initial_resampled,
                final_transform, final_dice_per_label, output_dir,
                save_visualization
            )
        
        self.final_transform = final_transform
        self.registration_method = registration
        
        return resampled_seg, final_transform
    
    def _normalize_image_ncc(self, image: sitk.Image) -> sitk.Image:
        """将图像归一化到0-1"""
        array = sitk.GetArrayFromImage(image)
        
        # 归一化到0-1
        array_min = np.min(array)
        array_max = np.max(array)
        array_norm = (array - array_min) / (array_max - array_min + 1e-8)
        
        # 转回SimpleITK
        image_norm = sitk.GetImageFromArray(array_norm.astype(np.float32))
        image_norm.CopyInformation(image)
        
        return image_norm

    def _compute_multi_label_distance_map(self, seg_image: sitk.Image) -> sitk.Image:
        """
        计算多标签的加权距离场
        
        将每个标签转换为带符号距离图，然后加权组合
        """
        seg_array = sitk.GetArrayFromImage(seg_image)
        distance_map = np.zeros_like(seg_array, dtype=np.float32)
        
        for label, weight in self.label_weights.items():
            # 创建当前标签的二值mask
            label_mask = (seg_array == label).astype(np.uint8)
            
            if np.sum(label_mask) == 0:
                continue
            
            # 转换为SimpleITK图像
            mask_img = sitk.GetImageFromArray(label_mask)
            mask_img.CopyInformation(seg_image)
            
            # 计算带符号距离图
            signed_distance = sitk.SignedMaurerDistanceMap(
                mask_img,
                insideIsPositive=False,
                squaredDistance=False,
                useImageSpacing=True
            )
            
            distance_array = sitk.GetArrayFromImage(signed_distance)
            
            # 加权累加
            distance_map += distance_array * weight
        
        # 转回SimpleITK
        distance_img = sitk.GetImageFromArray(distance_map)
        distance_img.CopyInformation(seg_image)
        
        return distance_img
    
    def _compute_centroid(self, seg_image: sitk.Image) -> Tuple[float, float]:
        """计算分割图的加权质心"""
        array = sitk.GetArrayFromImage(seg_image)
        
        # 只考虑前景（非零像素）
        mask = array > 0
        
        if not np.any(mask):
            # 如果没有前景，返回图像中心
            return (array.shape[1] / 2.0, array.shape[0] / 2.0)
        
        y_indices, x_indices = np.where(mask)
        
        # 计算质心
        centroid_x = np.mean(x_indices)
        centroid_y = np.mean(y_indices)
               
        centroid_physical = np.array([
            centroid_x,
            centroid_y]
        )
        
        return centroid_physical
    
    def _print_transform_parameters(self, transform, stage_name: str):
        """打印变换参数"""
        print(f"\n{stage_name}变换参数:")
        
        if isinstance(transform, sitk.Similarity2DTransform):
            print(f"  缩放: {transform.GetScale():.4f}")
            print(f"  旋转: {np.degrees(transform.GetAngle()):.2f}°")
            print(f"  平移: ({transform.GetTranslation()[0]:.2f}, {transform.GetTranslation()[1]:.2f})")
        elif isinstance(transform, sitk.Euler2DTransform):
            print(f"  旋转: {np.degrees(transform.GetAngle()):.2f}°")
            print(f"  平移: ({transform.GetTranslation()[0]:.2f}, {transform.GetTranslation()[1]:.2f})")
    
    def _save_results(
        self,
        fixed_seg, moving_seg, resampled_seg, initial_resampled,
        final_transform, dice_per_label, output_dir, save_visualization
    ):
        """保存配准结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n8. 保存结果...")
        
        # 保存配准后的分割图
        registered_path = output_path / "registered_segmentation.nii.gz"
        sitk.WriteImage(resampled_seg, str(registered_path))
        print(f"  配准分割: {registered_path}")
        
        # 保存Dice指标
        dice_df = pd.DataFrame([dice_per_label])
        dice_df['weighted_mean'] = self.compute_weighted_dice(fixed_seg, resampled_seg)
        dice_csv = output_path / "dice_scores.csv"
        dice_df.to_csv(dice_csv, index=False)
        print(f"  Dice分数: {dice_csv}")
        
        # 可视化
        if save_visualization:
            self._visualize_results(
                fixed_seg, moving_seg, initial_resampled, resampled_seg,
                output_path
            )
    
    def _visualize_results(
        self, fixed_seg, moving_seg, initial_resampled, resampled_seg, output_path
    ):
        """可视化配准结果"""
        print("\n9. 生成可视化...")
        
        fixed_array = sitk.GetArrayFromImage(fixed_seg).squeeze()
        moving_array = sitk.GetArrayFromImage(moving_seg).squeeze()
        initial_array = sitk.GetArrayFromImage(initial_resampled).squeeze()
        resampled_array = sitk.GetArrayFromImage(resampled_seg).squeeze()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行：分割图
        axes[0, 0].imshow(fixed_array, cmap='tab10', vmin=0, vmax=10)
        axes[0, 0].set_title('Fixed Segmentation (CT)', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(moving_array, cmap='tab10', vmin=0, vmax=10)
        axes[0, 1].set_title('Moving Segmentation (TEE) - Before', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(resampled_array, cmap='tab10', vmin=0, vmax=10)
        axes[0, 2].set_title('Moving Segmentation - After', fontsize=12)
        axes[0, 2].axis('off')
        
        # 第二行：叠加对比
        overlay_before = self._create_segmentation_overlay(fixed_array, initial_array)
        axes[1, 0].imshow(overlay_before)
        axes[1, 0].set_title('Overlay - Before Registration', fontsize=12)
        axes[1, 0].axis('off')
        
        overlay_after = self._create_segmentation_overlay(fixed_array, resampled_array)
        axes[1, 1].imshow(overlay_after)
        axes[1, 1].set_title('Overlay - After Registration', fontsize=12)
        axes[1, 1].axis('off')
        
        # Dice收敛曲线
        if self.dice_values:
            axes[1, 2].plot(np.arange(len(self.dice_values)) * 10, self.dice_values, 'b-', linewidth=2)
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('Weighted Dice Coefficient')
            axes[1, 2].set_title('Dice Convergence')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_ylim([0, 1])
        
        plt.tight_layout()
        vis_path = output_path / "registration_result.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  可视化: {vis_path}")
    
    def _create_segmentation_overlay(self, fixed_array, moving_array):
        """创建分割图叠加（不同颜色显示）"""
        overlay = np.zeros((*fixed_array.shape, 3))
        
        # 为不同标签分配颜色
        colors = {
            420: [1, 0, 0],     # 红色 - LA
            500: [0, 1, 0],     # 绿色 - LV
            550: [0, 0, 1],     # 蓝色 - RA
            600: [1, 1, 0]      # 黄色 - RV
        }
        
        # 固定图像：半透明
        for label, color in colors.items():
            mask = (fixed_array == label)
            for c in range(3):
                overlay[:, :, c] += mask * color[c] * 0.5
        
        # 移动图像：轮廓
        for label, color in colors.items():
            mask = (moving_array == label).astype(np.uint8)
            if np.any(mask):
                from scipy.ndimage import binary_erosion
                eroded = binary_erosion(mask)
                contour = mask & ~eroded
                for c in range(3):
                    overlay[:, :, c] += contour * color[c] * 0.8
        
        return np.clip(overlay, 0, 1)


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("CT-TEE 2D分割图配准（Dice优化）")
    print("=" * 60)
    
    # 配置路径
    moving_seg_path = r"D:\dataset\TEECT_data\ct\ct_train_1004_label_remapped\slice_062_t0.0_rx0_ry0.nii.gz"
    fixed_seg_path = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_label.nii.gz"
    output_dir = r"D:\dataset\TEECT_data\registration_results\segmentation_based"
    
    # 设置标签权重（可选）
    label_weights = {
        2: 1.0,  # LA
        1: 1.5,  # LV - 更重要
        3: 1.0,  # RA
        4: 1.0   # RV
    }
    
    # 执行配准
    registrator = SegmentationBasedRegistration2D(label_weights=label_weights)
    
    try:
        registered_seg, final_transform = registrator.register(
            fixed_seg_path=fixed_seg_path,
            moving_seg_path=moving_seg_path,
            output_dir=output_dir,
            number_of_iterations=500,
            use_scaling=False,  # 是否使用缩放
            save_visualization=True
        )
        
        print("\n✓ 配准成功完成！")
        
    except Exception as e:
        print(f"\n✗ 配准失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()