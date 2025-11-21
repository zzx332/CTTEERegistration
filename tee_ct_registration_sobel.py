import SimpleITK as sitk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os 
from datetime import datetime
import pandas as pd

class RigidRegistration2D:
    """
    基于互信息的2D图像刚性配准
    
    功能：
    - 支持多模态配准（CT和TEE超声）
    - 使用互信息作为相似性度量
    - 刚性变换（平移 + 旋转）
    - 支持NII.GZ格式
    """
    
    def __init__(self):
        """初始化配准器"""
        self.registration_method = None
        self.final_transform = None
        self.metric_values = []
        
    def register(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        output_dir: str = None,
        initial_transform: Optional[sitk.Transform] = None,
        number_of_iterations: int = 500,
        learning_rate: float = 1.0,
        min_step: float = 0.001,
        relaxation_factor: float = 0.5,
        sampling_percentage: float = 0.5,
        save_visualization: bool = True
    ) -> Tuple[sitk.Image, sitk.Transform]:
        """
        执行2D刚性配准
        
        Args:
            fixed_image_path: 固定图像路径（参考图像，通常是CT）
            moving_image_path: 移动图像路径（待配准图像，通常是TEE）
            output_dir: 输出目录
            initial_transform: 初始变换（可选）
            number_of_iterations: 最大迭代次数
            learning_rate: 学习率
            min_step: 最小步长
            relaxation_factor: 松弛因子
            sampling_percentage: 采样百分比（0-1）
            save_visualization: 是否保存可视化结果
            
        Returns:
            (配准后的图像, 最终变换)
        """
        print("=" * 60)
        print("2D刚性配准 - 基于互信息")
        print("=" * 60)
        
        # 读取图像
        print("\n1. 读取图像...")
        fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
        
        print(f"  固定图像: {fixed_image.GetSize()}")
        print(f"  移动图像: {moving_image.GetSize()}")
        print(f"  固定图像spacing: {fixed_image.GetSpacing()}")
        print(f"  移动图像spacing: {moving_image.GetSpacing()}")
        # 预处理：去除极端像素值
        print("  预处理ct图像...")
        #使用互信息配准时，需要预处理图像
        # fixed_image = self._preprocess_image(fixed_image, clip_percentiles=(0.5, 99.5))
        # moving_image = self._preprocess_image(moving_image, clip_percentiles=(0.5, 99.5))

        # 将超声图像转换为左手坐标系（左右翻转）
        print("  转换超声图像坐标系...")
        fixed_image = sitk.Flip(fixed_image, flipAxes=[True, True])  # X轴和Y轴都翻转
        fixed_image.SetOrigin((0,0))
        print("    已转换为左手坐标系（X轴翻转, Y轴翻转）")
        
        # 初始化配准方法
        print("\n2. 初始化配准方法...")
        # 在配准前应用
        print("  计算CT超声梯度幅值...")
        # fixed_gradient_magnitude = self._compute_gradient_magnitude(fixed_image)
        # moving_gradient_magnitude = self._compute_gradient_magnitude(moving_image)
        registration = sitk.ImageRegistrationMethod()
        fixed_mask = self.create_mask_from_intensity(fixed_image, threshold=0)
        moving_mask = self.create_mask_from_intensity(moving_image, threshold=-1000)
        # 设置mask
        # registration.SetMetricFixedMask(fixed_mask)
        # registration.SetMetricMovingMask(moving_mask)

        
        # 先归一化图像
        # fixed_image = self._normalize_image_ncc(fixed_image)
        # moving_image = self._normalize_image_ncc(moving_image)
        # fixed_image = self._normalize_image_ncc(fixed_image)

        # 使用NCC
        # registration.SetMetricAsMeanSquares()
        # registration.SetMetricAsCorrelation()
        # 设置相似性度量：互信息
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sampling_percentage)
        # 设置插值器
        registration.SetInterpolator(sitk.sitkLinear)
        # 使用Amoeba优化器（邻域搜索）
        # registration.SetOptimizerAsAmoeba(
        #     simplexDelta=5.0,
        #     # simplexDelta=2.0,
        #     numberOfIterations=500,
        #     parametersConvergenceTolerance=0.01,
        #     functionConvergenceTolerance=0.001,
        #     withRestarts=False
        # )
        # 设置优化器：梯度下降
        # registration.SetOptimizerAsRegularStepGradientDescent(
        #     learningRate=learning_rate,
        #     minStep=min_step,
        #     numberOfIterations=number_of_iterations,
        #     relaxationFactor=relaxation_factor
        # )
        registration.SetOptimizerAsPowell(
            numberOfIterations=500,
            maximumLineIterations=100,
            stepLength=1.0,
            stepTolerance=0.001,
            valueTolerance=1e-6
        )
        # registration.SetOptimizerAsConjugateGradientLineSearch(
        #     learningRate=0.1,
        #     numberOfIterations=500,
        #     convergenceMinimumValue=1e-6,
        #     convergenceWindowSize=10
        # )       
        # #穷举法
        # registration.SetOptimizerAsExhaustive(
        #     numberOfSteps=[10, 10, 10],  # 搜索旋转：10步，搜索平移：10步，搜索缩放：10步
        #     stepLength= 1.0     
        # )
        # moving_image = fixed_image
        show_initial_alignment = True 
        # 初始化变换
        if initial_transform is None:
            print("\n初始变换（图像中心对齐）:")
            fixed_center = fixed_image.GetOrigin() + (self.compute_image_centroid(fixed_image)) * fixed_image.GetSpacing()
            moving_center = moving_image.GetOrigin() + (self.compute_image_centroid(moving_image)) * moving_image.GetSpacing()
            # fixed_center = fixed_image.GetOrigin() + np.array(fixed_image.GetSize()) * fixed_image.GetSpacing() / 2
            # moving_center = moving_image.GetOrigin() + np.array(moving_image.GetSize()) * moving_image.GetSpacing() / 2
            # translation = (fixed_center[0] - moving_center[0], fixed_center[1] - moving_center[1])
            translation = (moving_center[0] - fixed_center[0], moving_center[1] - fixed_center[1])
            # initial_transform = sitk.CompositeTransform(2)
            
            # trans_tf = sitk.TranslationTransform(2)
            # trans_tf.SetOffset([translation[0], translation[1]])
            # initial_transform.AddTransform(trans_tf)
            
            # rot_tf = sitk.Euler2DTransform()
            # rot_tf.SetCenter(moving_center)
            # rot_tf.SetAngle(0)
            # initial_transform.AddTransform(rot_tf)

            # initial_transform = sitk.Euler2DTransform()
            # initial_transform.SetCenter(moving_center)
            # # translation = (np.random.rand(2) * 2 - 1) * 30
            # initial_transform.SetTranslation(translation)
            # initial_transform.SetAngle(0)
            # initial_transform.SetAngle(np.pi/6)
            #加入放缩
            # initial_transform = sitk.Similarity2DTransform()
            # initial_transform.SetCenter(moving_center)
            # initial_transform.SetTranslation(translation)
            # initial_transform.SetAngle(0)
            # initial_transform.SetScale(1.0)  # 初始缩放为1.0（无缩放）
            # # 使用简单的平移变换
            initial_transform = sitk.Euler2DTransform()
            # initial_transform.SetCenter(fixed_center)
            initial_transform.SetCenter(moving_center)
            initial_transform.SetTranslation(translation)
            initial_transform.SetAngle(0)
            
            print(f"  初始平移: ({translation[0]:.2f}, {translation[1]:.2f}) mm")
            
            # # 应用初始变换并可视化
            # print("  生成初始对齐可视化...")
        initial_resampled = sitk.Resample(
            moving_image, fixed_image, initial_transform,
            sitk.sitkLinear, 0.0, moving_image.GetPixelID()
        )
        
        if show_initial_alignment:
            # 创建叠加图
            fixed_array = sitk.GetArrayFromImage(fixed_image).squeeze()
            initial_array = sitk.GetArrayFromImage(initial_resampled).squeeze()
            
            # 归一化
            def norm(img):
                p1, p99 = np.percentile(img, [1, 99])
                return np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)
            
            # 创建叠加（灰色固定图像 + 绿色移动图像）
            # overlay = np.stack([norm(fixed_array)] * 3, axis=2)
            overlay = np.zeros((*fixed_array.shape, 3))
            overlay[:, :, 0] = norm(fixed_array)
            overlay[:, :, 1] = np.clip(overlay[:, :, 1] + norm(initial_array) * 0.6, 0, 1)
            
            # 显示
            plt.figure(figsize=(10, 10))
            plt.imshow(overlay)
            plt.title('Initial Alignment (Gray=Fixed, Green=Moving)', fontsize=14)
            plt.axis('off')
            
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path / "initial_alignment.png", dpi=150, bbox_inches='tight')
                print(f"  已保存: {output_path / 'initial_alignment.png'}")
            
            # plt.show()
            plt.close()

        registration.SetInitialTransform(initial_transform, inPlace=False)
        # registration.SetOptimizerScales([10, 1.0, 1.0])
        registration.SetOptimizerScales([5, 0.95, 0.1, 0.1])
        # registration.SetOptimizerScalesFromIndexShift()
        # registration.SetOptimizerScalesFromPhysicalShift()
        # 设置多分辨率策略
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        # registration.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
        # registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
        # registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # 添加观察器以监控配准过程
        self.metric_values = []
        registration.AddCommand(sitk.sitkIterationEvent, 
                               lambda: self._iteration_callback(registration))
        
        print("\n3. 开始配准...")
        print(f"  优化器: Regular Step Gradient Descent")
        print(f"  相似性度量: Mattes Mutual Information")
        print(f"  最大迭代次数: {number_of_iterations}")
        print(f"  学习率: {learning_rate}")
        print(f"  采样率: {sampling_percentage * 100}%")
        
        # 执行配准
        try:
            final_transform = registration.Execute(fixed_image, moving_image)
            
            print("\n4. 配准完成!")
            print(f"  优化器停止条件: {registration.GetOptimizerStopConditionDescription()}")
            print(f"  最终度量值: {registration.GetMetricValue():.6f}")
            print(f"  实际迭代次数: {len(self.metric_values)}")
            
        except Exception as e:
            print(f"\n配准失败: {e}")
            raise
        
        # 保存最终变换
        self.final_transform = final_transform
        self.registration_method = registration
        
        # 应用变换
        print("\n5. 应用变换...")
        resampled_image = sitk.Resample(
            moving_image,
            fixed_image,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID()
        )
        
        # 打印变换参数
        print("初始变换参数:")
        self._print_transform_parameters(initial_transform)
        print("最终变换参数:")
        self._print_transform_parameters(final_transform)
        
        # 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            print("\n6. 保存结果...")
            # 保存配准后的图像
            registered_path = output_path / "registered_image.nii.gz"
            sitk.WriteImage(resampled_image, str(registered_path))
            print(f"  配准图像: {registered_path}")
            
            # # 保存变换
            # transform_path = output_path / "transform.tfm"
            # sitk.WriteTransform(final_transform, str(transform_path))
            # print(f"  变换文件: {transform_path}")
            
            # 保存可视化结果
            if save_visualization:
                self._save_visualization(
                    fixed_image, 
                    initial_resampled, 
                    resampled_image, 
                    output_path
                )
        
        print("\n" + "=" * 60)
        
        return resampled_image, final_transform
    
    def _iteration_callback(self, registration_method):
        """迭代回调函数，用于监控配准过程"""
        metric_value = registration_method.GetMetricValue()
        self.metric_values.append(metric_value)
        
        # 每10次迭代打印一次
        if len(self.metric_values) % 10 == 0:
            print(f"  迭代 {len(self.metric_values)}: 度量值 = {metric_value:.6f}")
    
    def _print_transform_parameters(self, transform):
        """打印变换参数"""
        
        if isinstance(transform, sitk.Similarity2DTransform):
            # Similarity2DTransform: 缩放 + 旋转 + 平移
            angle_deg = np.degrees(transform.GetAngle())
            translation = transform.GetTranslation()
            scale = transform.GetScale()
            center = transform.GetCenter()
            
            print(f"  缩放: {scale:.4f}")
            print(f"  旋转角度: {angle_deg:.2f}°")
            print(f"  平移: ({translation[0]:.2f}, {translation[1]:.2f}) mm")
            print(f"  旋转中心: ({center[0]:.2f}, {center[1]:.2f})")
            
        elif isinstance(transform, sitk.Euler2DTransform):
            # Euler2DTransform: 旋转 + 平移
            angle_deg = np.degrees(transform.GetAngle())
            translation = transform.GetTranslation()
            center = transform.GetCenter()
            
            print(f"  旋转角度: {angle_deg:.2f}°")
            print(f"  平移: ({translation[0]:.2f}, {translation[1]:.2f}) mm")
            print(f"  旋转中心: ({center[0]:.2f}, {center[1]:.2f})")
            
        elif isinstance(transform, sitk.CompositeTransform):
            print(f"  复合变换，包含 {transform.GetNumberOfTransforms()} 个变换")
            for i in range(transform.GetNumberOfTransforms()):
                print(f"  变换 {i+1}:")
                self._print_transform_parameters(transform.GetNthTransform(i))    # 在读取图像后添加归一化
    
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

    def create_mask_from_intensity(self, image: sitk.Image, threshold: float = 0) -> sitk.Image:
        """
        根据强度阈值创建mask
        
        Args:
            image: 输入图像
            threshold: 强度阈值，大于此值的像素被认为是前景
        
        Returns:
            二值mask图像
        """
        # 转换为numpy数组
        array = sitk.GetArrayFromImage(image)
        nrows, ncols = array.shape
        mask_array = np.ones((nrows, ncols), dtype=np.uint8)
        for i in range(nrows):
            l = 0
            r = ncols - 1
            while l < r and (array[i, l] == threshold or array[i, r] == threshold):
                if array[i, l] == threshold:
                    mask_array[i, l] = 0
                    l += 1
                if array[i, r] == threshold:
                    mask_array[i, r] = 0
                    r -= 1
        mask_image = sitk.GetImageFromArray(mask_array)
        mask_image.CopyInformation(image)
        return mask_image
    def _compute_gradient_magnitude(self, image: sitk.Image) -> sitk.Image:
        """
        使用Sobel算子计算梯度幅值
        
        Args:
            image: 输入图像
        
        Returns:
            梯度幅值图像
        """
        # 使用SimpleITK的梯度幅值滤波器
        gradient_filter = sitk.GradientMagnitudeImageFilter()
        gradient_image = gradient_filter.Execute(image)
        
        return gradient_image

    def _save_visualization(
        self, 
        fixed_image: sitk.Image, 
        moving_image: sitk.Image,
        resampled_image: sitk.Image,
        output_path: Path
    ):
        """保存可视化结果"""
        print("\n7. 生成可视化...")
        
        # 转换为numpy数组
        fixed_array = sitk.GetArrayFromImage(fixed_image).squeeze()
        moving_array = sitk.GetArrayFromImage(moving_image).squeeze()
        resampled_array = sitk.GetArrayFromImage(resampled_image).squeeze()
        # 将moving_image重采样到fixed_image的空间（用于配准前对比）
        identity_transform = sitk.Transform(2, sitk.sitkIdentity)
        moving_resampled_for_comparison = sitk.Resample(
            moving_image,
            fixed_image,
            identity_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID()
        )
        moving_array_resized = sitk.GetArrayFromImage(moving_resampled_for_comparison).squeeze()
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 第一行：单独显示
        axes[0, 0].imshow(fixed_array, cmap='gray')
        axes[0, 0].set_title('Fixed Image (CT)', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(moving_array, cmap='gray')
        axes[0, 1].set_title('Moving Image (TEE) - Before', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(resampled_array, cmap='gray')
        axes[0, 2].set_title('Moving Image (TEE) - After', fontsize=12)
        axes[0, 2].axis('off')
        
        # 第二行：叠加显示
        # 配准前叠加（红-绿）
        overlay_before = self._create_overlay(fixed_array, moving_array_resized)
        axes[1, 0].imshow(overlay_before)
        axes[1, 0].set_title('Overlay Before (Red=Fixed, Green=Moving)', fontsize=12)
        axes[1, 0].axis('off')
        
        # 配准后叠加（红-绿）
        overlay_after = self._create_overlay(fixed_array, resampled_array)
        axes[1, 1].imshow(overlay_after)
        axes[1, 1].set_title('Overlay After (Red=Fixed, Green=Moving)', fontsize=12)
        axes[1, 1].axis('off')
        
        # 差异图
        difference = np.abs(fixed_array - resampled_array)
        axes[1, 2].imshow(difference, cmap='hot')
        axes[1, 2].set_title('Absolute Difference', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存
        vis_path = output_path / "registration_result.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        # plt.show()
        # plt.close()
        print(f"  可视化结果: {vis_path}")
        
        # 保存度量值曲线
        if self.metric_values:
            self._plot_metric_curve(output_path)
    
    def _create_overlay(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """创建红-绿叠加图像"""
        # 归一化到0-1
        img1_norm = self._normalize_image(image1)
        img2_norm = self._normalize_image(image2)
        
        # 创建RGB图像
        overlay = np.zeros((*img1_norm.shape, 3))
        overlay[:, :, 0] = img1_norm  # 红色通道：固定图像
        overlay[:, :, 1] = img2_norm  # 绿色通道：移动图像
        
        return overlay
    def _create_overlay_gray(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """创建叠加图像：灰色固定图像 + 绿色移动图像"""
        # 归一化到0-1
        img1_norm = self._normalize_image(image1)
        img2_norm = self._normalize_image(image2)
        
        # 创建RGB图像
        overlay = np.zeros((*img1_norm.shape, 3))
        
        # 固定图像显示为灰色（RGB三通道相同）
        overlay[:, :, 0] = img1_norm  # R通道
        overlay[:, :, 1] = img1_norm  # G通道


        
        overlay[:, :, 2] = img1_norm  # B通道
        
        # 叠加绿色移动图像（使用加法混合，并限制在0-1范围）
        overlay[:, :, 1] = np.clip(overlay[:, :, 1] + img2_norm * 0.7, 0, 1)  # 绿色通道叠加
        
        return overlay
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """归一化图像到0-1"""
        image = image.astype(float)
        img_min = np.percentile(image, 1)
        img_max = np.percentile(image, 99)
        
        if img_max > img_min:
            image = np.clip(image, img_min, img_max)
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)
        
        return image

    def compute_image_centroid(self, image: sitk.Image) -> np.ndarray:
        array = sitk.GetArrayFromImage(image)
    
        y_indices, x_indices = np.indices(array.shape)
        total_intensity = np.sum(array)
        centroid_y = np.sum(y_indices * array) / total_intensity
        centroid_x = np.sum(x_indices * array) / total_intensity
        
        centroid_voxel = np.array([centroid_x, centroid_y])
        return centroid_voxel

    def _plot_metric_curve(self, output_path: Path):
        """绘制度量值曲线"""
        if not self.metric_values:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.metric_values, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Metric Value (Mutual Information)', fontsize=12)
        plt.title('Registration Convergence', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 标注最终值
        final_value = self.metric_values[-1]
        plt.axhline(y=final_value, color='r', linestyle='--', alpha=0.5)
        plt.text(len(self.metric_values) * 0.5, final_value, 
                f'Final: {final_value:.6f}', 
                fontsize=10, color='r', va='bottom')
        
        plt.tight_layout()
        metric_path = output_path / "metric_curve.png"
        plt.savefig(metric_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  度量曲线: {metric_path}")

    def _preprocess_image(self, image: sitk.Image, clip_percentiles: tuple = (0.5, 99.5)) -> sitk.Image:
        """
        预处理图像：裁剪极端像素值
        
        Args:
            image: 输入图像
            clip_percentiles: 裁剪百分位数 (lower, upper)
        
        Returns:
            处理后的图像
        """
        # 转换为numpy数组
        array = sitk.GetArrayFromImage(image)
        
        # 计算百分位数
        # lower = np.percentile(array, clip_percentiles[0])
        lower = -1000
        upper = np.percentile(array, clip_percentiles[1])
        
        # 裁剪
        array_clipped = np.clip(array, lower, upper)
        
        # 转回SimpleITK图像并保留元数据
        image_clipped = sitk.GetImageFromArray(array_clipped)
        image_clipped.CopyInformation(image)
        
        print(f"    强度裁剪: [{lower:.2f}, {upper:.2f}]")
        
        return image_clipped

    def apply_transform_to_file(
        self,
        moving_image_path: str,
        fixed_image_path: str,
        transform: sitk.Transform,
        output_path: str
    ):
        """
        将已有的变换应用到新的图像
        
        Args:
            moving_image_path: 待变换的图像
            fixed_image_path: 参考图像（定义输出空间）
            transform: 变换对象
            output_path: 输出路径
        """
        moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
        fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
        
        resampled = sitk.Resample(
            moving_image,
            fixed_image,
            transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID()
        )
        
        sitk.WriteImage(resampled, output_path)
        print(f"变换已应用，结果保存至: {output_path}")


def main():
    """
    主函数示例
    """
    print("=" * 60)
    print("CT-TEE 2D图像刚性配准")
    print("=" * 60)
    
    # ===== 配置路径 =====
    # moving_image_path = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge\slice_062_t0.0_rx0_ry0_edge.nii.gz"
    moving_image_path = r"D:\dataset\TEECT_data\ct\ct_train_1004_image\slice_062_t0.0_rx0_ry0.nii.gz"
    # moving_image_path = r"D:\dataset\TEECT_data\ct\ct_train_1004_heatmap\slice_062_t0.0_rx0_ry0_heatmap_heatmap.nii.gz"
    # moving_image_path = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge_individual\slice_062_t0.0_rx0_ry0_multi_edge.nii.gz"
    # # moving_image_path = r"D:\dataset\TEECT_data\ct\ct_train_1002_image\slice_067_t0.0_rx25_ry0.nii.gz"
    # # moving_image_path = r"D:\dataset\TEECT_data\ct\ct_train_1002_image\slice_115_t5.0_rx25_ry-50.nii.gz"
    # fixed_image_path = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_label_heatmap_heatmap.nii.gz"
    # fixed_image_path = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_label_multi_edge.nii.gz"
    fixed_image_path = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_image.nii.gz"
    output_dir = r"D:\dataset\TEECT_data\registration_results\patient-1-4"
    # moving_image_path = r"D:\dataset\TEECT_data\ct\brain\slice_070_t0.0_rx50_ry-50.nii.gz"
    # fixed_image_path = r"D:\dataset\TEECT_data\mr\brain\slice_062_t0.0_rx0_ry0.nii.gz"
    # output_dir = r"D:\dataset\TEECT_data\registration_results_brain1"
    transform = sitk.ReadTransform(r"D:\dataset\TEECT_data\registration_results\icp\slice_062_t0.0_rx0_ry0_edge.nii_transform.tfm")
    # ===== 配准参数 =====
    config = {
        'number_of_iterations': 500,      # 最大迭代次数
        'learning_rate': 1.0,             # 学习率
        'min_step': 0.01,                # 最小步长
        'relaxation_factor': 0.5,         # 松弛因子
        'sampling_percentage': 0.5,       # 采样百分比（20%的像素用于计算度量）
        'save_visualization': True        # 保存可视化结果
    }
    
    # ===== 执行配准 =====
    registrator = RigidRegistration2D()
    
    try:
        registered_image, final_transform = registrator.register(
            fixed_image_path=fixed_image_path,
            moving_image_path=moving_image_path,
            output_dir=output_dir,
            initial_transform=transform,
            **config
        )
        
        print("\n配准成功完成！")
        
    except Exception as e:
        print(f"\n配准失败: {e}")
        import traceback
        traceback.print_exc()


def batch_registration():
    """
    批量配准示例
    """
    print("=" * 60)
    print("批量配准")
    print("=" * 60)
    moving_image_dir = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge_individual"
    fixed_image_path = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_label_multi_edge.nii.gz"
    moving_image_list = os.listdir(moving_image_dir)
    output_dir = r"D:\dataset\TEECT_data\registration_results\patient-1-4_boundary"
    registrator = RigidRegistration2D()
    # ===== 配准参数 =====
    config = {
        'number_of_iterations': 500,      # 最大迭代次数
        'learning_rate': 1.0,             # 学习率
        'min_step': 0.01,                # 最小步长
        'relaxation_factor': 0.5,         # 松弛因子
        'sampling_percentage': 0.5,       # 采样百分比（20%的像素用于计算度量）
        'save_visualization': True        # 保存可视化结果
    }
    results = []
    for i, moving_image in enumerate(moving_image_list):
        if i % 2 == 0:
            continue
        print(f"\n处理第{i+1}个moving image: {moving_image.split('.nii.gz')[0]}")
        
        try:
            start_time = datetime.now()
            registered_image, transform = registrator.register(
                fixed_image_path=fixed_image_path,
                moving_image_path=os.path.join(moving_image_dir, moving_image),
                output_dir=os.path.join(output_dir, moving_image.split('.nii.gz')[0]),
                 **config
            )
            end_time = datetime.now()
                        
            # 计算配准耗时
            elapsed_time = (end_time - start_time).total_seconds()
            print(f"配准对 {i+1} 完成，耗时: {elapsed_time}秒")
                        # 获取配准指标
            if hasattr(registrator, 'registration_method') and registrator.registration_method:
                final_metric = registrator.registration_method.GetMetricValue()
            else:
                final_metric = None
            
            # 获取变换参数
            if isinstance(transform, sitk.Euler2DTransform):
                angle_deg = np.degrees(transform.GetAngle())
                translation = transform.GetTranslation()
                tx, ty = translation[0], translation[1]
                center = transform.GetCenter()
            else:
                angle_deg = None
                tx, ty = None, None
                center = (None, None)
            
            # 记录结果
            result = {
                'pair_id': moving_image.split('.nii.gz')[0],
                'patient': 'patient-1-4',
                'fixed_slice': 'slice_060_image.nii.gz',
                'moving_slice': moving_image,
                'status': 'success',
                'final_metric': final_metric,
                'rotation_angle_deg': angle_deg,
                'translation_x_mm': tx,
                'translation_y_mm': ty,
                'rotation_center_x': center[0] if center[0] is not None else None,
                'rotation_center_y': center[1] if center[1] is not None else None,
                'elapsed_time_sec': elapsed_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error_message': None
            }
            
            print(f"\n✓ 配准对 {moving_image.split('.nii.gz')[0]} 完成")
            print(f"  最终度量值: {final_metric:.6f}" if final_metric else "  度量值: N/A")
            print(f"  旋转角度: {angle_deg:.2f}°" if angle_deg is not None else "  旋转角度: N/A")
            print(f"  平移: ({tx:.2f}, {ty:.2f}) mm" if tx is not None else "  平移: N/A")
            print(f"  耗时: {elapsed_time:.1f}秒")
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # 记录失败结果
            result = {
                'pair_id': moving_image.split('.nii.gz')[0],
                'patient': 'patient-1-4',
                'fixed_slice': 'slice_060_image.nii.gz',
                'moving_slice': moving_image,
                'status': 'failed',
                'final_metric': None,
                'rotation_angle_deg': None,
                'translation_x_mm': None,
                'translation_y_mm': None,
                'rotation_center_x': None,
                'rotation_center_y': None,
                'elapsed_time_sec': elapsed_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error_message': str(e)
            }
            
            print(f"\n✗ 配准对 {moving_image.split('.nii.gz')[0]} 失败: {e}")
        results.append(result)
        # 创建DataFrame
    df = pd.DataFrame(results)
    

    dir_name = output_dir.split('\\')[-1]
    output_csv = os.path.join(output_dir, f"{dir_name}.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到CSV: {output_csv}")



if __name__ == "__main__":
    # 运行单次配准
    main()
    
    # 或运行批量配准
    # batch_registration()
# 
