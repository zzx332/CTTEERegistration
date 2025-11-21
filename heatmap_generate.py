"""
从二值边缘图生成高斯概率热图
用于更鲁棒的图像配准
"""

from re import T
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from scipy import ndimage
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def generate_gaussian_heatmap(
    edge_image: np.ndarray,
    sigma: float = 3.0,
    normalize: bool = True,
    max_value: float = 1.0
) -> np.ndarray:
    """
    从二值边缘图生成高斯概率热图
    
    Args:
        edge_image: 二值边缘图 (H, W)，边缘像素值为1或更大
        sigma: 高斯核的标准差（控制热图的扩散范围）
        normalize: 是否归一化到[0, max_value]
        max_value: 归一化后的最大值
    
    Returns:
        heatmap: 高斯概率热图 (H, W)
    """
    # 确保输入是浮点型
    edge_float = edge_image.astype(np.float32)
    
    # 对边缘图应用高斯滤波
    # 这会在边缘周围创建平滑的概率分布
    heatmap = ndimage.gaussian_filter(edge_float, sigma=sigma)
    
    # 归一化
    if normalize and heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max()) * max_value
    
    return heatmap


def generate_distance_transform_heatmap(
    edge_image: np.ndarray,
    max_distance: float = 10.0,
    invert: bool = True
) -> np.ndarray:
    """
    使用距离变换生成热图
    
    Args:
        edge_image: 二值边缘图
        max_distance: 最大距离（超过此距离的像素值为0）
        invert: 是否反转（True=边缘处值最大）
    
    Returns:
        heatmap: 距离变换热图
    """
    # 创建二值mask
    binary_edge = (edge_image > 0).astype(np.uint8)
    
    # 计算到最近边缘的距离
    distance = ndimage.distance_transform_edt(1 - binary_edge)
    
    # 限制最大距离
    distance = np.clip(distance, 0, max_distance)
    
    # 转换为概率（距离越近，概率越高）
    if invert:
        heatmap = 1.0 - (distance / max_distance)
    else:
        heatmap = distance / max_distance
    
    return heatmap.astype(np.float32)


def generate_multi_label_heatmap(
    multi_label_edge: np.ndarray,
    sigma: float = 3.0,
    label_weights: dict = None
) -> np.ndarray:
    """
    为多标签边缘图生成加权热图
    
    Args:
        multi_label_edge: 多标签边缘图，不同标签有不同的值
        sigma: 高斯核标准差
        label_weights: 标签权重字典，如 {1: 1.0, 2: 0.8}
    
    Returns:
        heatmap: 加权热图
    """
    unique_labels = np.unique(multi_label_edge)
    unique_labels = unique_labels[unique_labels > 0]  # 排除背景
    
    if label_weights is None:
        label_weights = {label: 1.0 for label in unique_labels}
    
    # 初始化热图
    heatmap = np.zeros_like(multi_label_edge, dtype=np.float32)
    
    # 为每个标签生成热图并加权
    for label in unique_labels:
        # 提取当前标签的边缘
        label_edge = (multi_label_edge == label).astype(np.float32)
        
        # 生成高斯热图
        label_heatmap = ndimage.gaussian_filter(label_edge, sigma=sigma)
        
        # 应用权重
        weight = label_weights.get(label, 1.0)
        label_heatmap *= weight
        
        # 累加到总热图
        heatmap = np.maximum(heatmap, label_heatmap)
    
    # 归一化
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def batch_generate_heatmaps(
    edge_dir: str,
    output_dir: str,
    sigma: float = 3.0,
    method: str = 'gaussian',
    visualize: bool = True
):
    """
    批量生成热图
    
    Args:
        edge_dir: 边缘图目录
        output_dir: 输出热图目录
        sigma: 高斯核标准差
        method: 生成方法 ('gaussian', 'distance', 'multi_label')
        visualize: 是否生成可视化
    """
    edge_dir = Path(edge_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有边缘图文件
    edge_files = sorted(edge_dir.glob("*label_edge*.nii*"))
    
    print(f"找到 {len(edge_files)} 个边缘图文件")
    print(f"生成方法: {method}")
    print(f"Sigma: {sigma}")
    print("=" * 60)
    
    for edge_path in edge_files:
        print(f"\n处理: {edge_path.name}")
        
        # 读取边缘图
        edge_img = sitk.ReadImage(str(edge_path))
        edge_array = sitk.GetArrayFromImage(edge_img)
        
        # 根据维度处理
        if edge_array.ndim == 3:
            # 3D图像，逐层处理
            heatmap_array = np.zeros_like(edge_array, dtype=np.float32)
            
            for slice_idx in range(edge_array.shape[0]):
                edge_slice = edge_array[slice_idx]
                
                if np.all(edge_slice == 0):
                    continue
                
                # 生成热图
                if method == 'gaussian':
                    heatmap_slice = generate_gaussian_heatmap(edge_slice, sigma=sigma)
                elif method == 'distance':
                    heatmap_slice = generate_distance_transform_heatmap(edge_slice)
                elif method == 'multi_label':
                    heatmap_slice = generate_multi_label_heatmap(edge_slice, sigma=sigma)
                else:
                    raise ValueError(f"未知方法: {method}")
                
                heatmap_array[slice_idx] = heatmap_slice
                
                print(f"  切片 {slice_idx}: 热图范围 [{heatmap_slice.min():.3f}, {heatmap_slice.max():.3f}]")
        
        else:  # 2D
            if method == 'gaussian':
                heatmap_array = generate_gaussian_heatmap(edge_array, sigma=sigma)
            elif method == 'distance':
                heatmap_array = generate_distance_transform_heatmap(edge_array)
            elif method == 'multi_label':
                heatmap_array = generate_multi_label_heatmap(edge_array, sigma=sigma)
            
            print(f"  热图范围: [{heatmap_array.min():.3f}, {heatmap_array.max():.3f}]")
        
        # 保存热图
        heatmap_img = sitk.GetImageFromArray(heatmap_array)
        heatmap_img.CopyInformation(edge_img)
        
        output_name = edge_path.name.replace('_edge', '_heatmap').replace('.nii.gz', '_heatmap.nii.gz')
        output_path = output_dir / output_name
        
        sitk.WriteImage(heatmap_img, str(output_path), useCompression=True)
        print(f"  ✓ 保存到: {output_path}")
        
        # 可视化
        if visualize:
            visualize_heatmap(edge_array, heatmap_array, output_dir, edge_path.stem)
    
    print(f"\n{'='*60}")
    print(f"✓ 完成！共处理 {len(edge_files)} 个文件")
    print(f"输出目录: {output_dir}")


def visualize_heatmap(
    edge: np.ndarray,
    heatmap: np.ndarray,
    output_dir: Path,
    base_name: str
):
    """
    可视化边缘图和热图的对比
    """
    # 如果是3D，取中间切片
    if edge.ndim == 3:
        mid_slice = edge.shape[0] // 2
        edge = edge[mid_slice]
        heatmap = heatmap[mid_slice]
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始边缘图
    axes[0].imshow(edge, cmap='gray')
    axes[0].set_title('Original Edge Map')
    axes[0].axis('off')
    
    # 热图
    im = axes[1].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Gaussian Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # 叠加
    axes[2].imshow(edge, cmap='gray', alpha=0.5)
    axes[2].imshow(heatmap, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    vis_path = output_dir / f"{base_name}_heatmap_vis.png"
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  可视化: {vis_path}")


def compare_sigma_values(
    edge_image: np.ndarray,
    sigmas: list = [1.0, 3.0, 5.0, 10.0],
    output_path: str = "sigma_comparison.png"
):
    """
    比较不同sigma值的效果
    """
    n_sigmas = len(sigmas)
    fig, axes = plt.subplots(2, n_sigmas, figsize=(4*n_sigmas, 8))
    
    for idx, sigma in enumerate(sigmas):
        # 生成热图
        heatmap = generate_gaussian_heatmap(edge_image, sigma=sigma)
        
        # 显示边缘图
        axes[0, idx].imshow(edge_image, cmap='gray')
        axes[0, idx].set_title(f'Edge (σ={sigma})')
        axes[0, idx].axis('off')
        
        # 显示热图
        im = axes[1, idx].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
        axes[1, idx].set_title(f'Heatmap (σ={sigma})')
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Sigma对比图已保存: {output_path}")


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数"""
    
    # 配置路径
    # edge_dir = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge"
    edge_dir = r"D:\dataset\TEECT_data\tee\patient-1-4"
    output_dir = r"D:\dataset\TEECT_data\tee\patient-1-4"
    
    # 批量生成热图
    batch_generate_heatmaps(
        edge_dir=edge_dir,
        output_dir=output_dir,
        sigma=10.0,              # 高斯核标准差（控制扩散范围）
        method='gaussian',   # 'gaussian', 'distance', 或 'multi_label'
        visualize=False          # 生成可视化
    )


def demo_sigma_comparison():
    """演示不同sigma值的效果"""
    
    # 加载一个示例边缘图
    edge_path = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge_individual\slice_062_t0.0_rx0_ry0_multi_edge.nii.gz"
    edge_img = sitk.ReadImage(edge_path)
    edge_array = sitk.GetArrayFromImage(edge_img)
    
    # 如果是3D，取中间切片
    if edge_array.ndim == 3:
        edge_array = edge_array[edge_array.shape[0] // 2]
    
    # 比较不同sigma
    compare_sigma_values(
        edge_array,
        sigmas=[1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
        output_path="sigma_comparison.png"
    )


if __name__ == "__main__":
    # 运行主函数
    main()
    
    # 或运行sigma对比演示
    # demo_sigma_comparison()