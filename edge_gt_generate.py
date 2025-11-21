"""
从四腔心分割标签生成边缘真值
用于训练结构化决策森林
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from scipy import ndimage
from typing import Tuple, Optional
import cv2


def generate_edge_from_segmentation(
    seg_array: np.ndarray,
    edge_type: str = 'boundary',
    dilate_size: int = 0  # 改为0，不膨胀
) -> np.ndarray:
    """
    从分割标签生成单像素宽的边缘标签
    
    Args:
        seg_array: 分割标签数组 (H, W)
        edge_type: 边缘类型
            - 'boundary': 不同区域之间的边界
            - 'contour': 每个区域的轮廓（推荐，更精细）
            - 'both': 两者结合
        dilate_size: 边缘膨胀大小（0=单像素，1=3像素宽）
    
    Returns:
        edge_map: 边缘标签 (H, W)，值为0或1
    """
    edge_map = np.zeros_like(seg_array, dtype=np.uint8)
    
    if edge_type in ['boundary', 'both']:
        # 方法1：使用形态学方法提取单像素边界
        # 腐蚀后与原图相减，得到单像素边界
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(seg_array.astype(np.uint8), kernel, iterations=1)
        
        # 边界 = 原始 - 腐蚀（这样得到的是单像素宽）
        boundary = ((seg_array > 0) & (eroded == 0)).astype(np.uint8)
        edge_map = np.maximum(edge_map, boundary)
    
    if edge_type in ['contour', 'both']:
        # 方法2：使用cv2.findContours提取精确轮廓（单像素）
        unique_labels = np.unique(seg_array)
        
        for label in unique_labels:
            if label == 0:  # 跳过背景
                continue
            
            # 二值化当前标签
            binary = (seg_array == label).astype(np.uint8)
            
            # 提取轮廓（CHAIN_APPROX_NONE保证所有轮廓点）
            contours, _ = cv2.findContours(
                binary, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_NONE  # 保留所有轮廓点，不压缩
            )
            
            # 绘制单像素宽的轮廓
            cv2.drawContours(edge_map, contours, -1, 1, thickness=1)
    
    # 可选：膨胀边缘（通常设为0保持单像素）
    if dilate_size > 0:
        kernel = np.ones((dilate_size * 2 + 1, dilate_size * 2 + 1), np.uint8)
        edge_map = cv2.dilate(edge_map, kernel, iterations=1)
    
    return edge_map


def generate_cardiac_chamber_edges(
    seg_array: np.ndarray,
    chamber_labels: dict = None,
    include_internal: bool = True
) -> np.ndarray:
    """
    专门针对四腔心分割生成边缘
    
    Args:
        seg_array: 分割标签 (H, W)
        chamber_labels: 腔室标签映射，如 {1: 'LV', 2: 'RV', 3: 'LA', 4: 'RA'}
        include_internal: 是否包含腔室内部的边缘（如乳头肌）
    
    Returns:
        edge_map: 边缘标签
    """
    if chamber_labels is None:
        # 默认四腔心标签
        chamber_labels = {
            1: 'LV',  # 左心室
            2: 'RV',  # 右心室
            3: 'LA',  # 左心房
            4: 'RA'   # 右心房
        }
    
    edge_map = np.zeros_like(seg_array, dtype=np.uint8)
    
    # 1. 提取每个腔室的边界
    for label_id, chamber_name in chamber_labels.items():
        if label_id not in seg_array:
            continue
        
        # 二值化当前腔室
        chamber_mask = (seg_array == label_id).astype(np.uint8)
        
        # 形态学操作：腐蚀
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(chamber_mask, kernel, iterations=1)
        
        # 边界 = 原始 - 腐蚀
        boundary = chamber_mask - eroded
        
        edge_map = np.maximum(edge_map, boundary)
    
    # 2. 如果不包含内部边缘，只保留腔室之间的边界
    if not include_internal:
        # 创建整体心脏mask
        heart_mask = (seg_array > 0).astype(np.uint8)
        
        # 腐蚀
        eroded_heart = cv2.erode(heart_mask, kernel, iterations=1)
        
        # 外边界
        outer_boundary = heart_mask - eroded_heart
        
        # 只保留外边界
        edge_map = np.minimum(edge_map, outer_boundary)
    
    return edge_map


def batch_generate_edge_labels(
    label_dir: str,
    output_dir: str,
    chamber_to_edge_map: dict,
    edge_type: str = 'contour',
    dilate_size: int = 1
):
    """
    批量生成多标签边缘图，每个心腔边缘有不同的值，保存在同一文件中。
    
    Args:
        label_dir: 分割标签目录
        output_dir: 输出边缘标签目录
        chamber_to_edge_map: 心腔分割ID到新边缘标签ID的映射
        edge_type: 边缘类型
        dilate_size: 边缘膨胀大小
    """
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = sorted(label_dir.glob("*.nii*"))
    # label_files = sorted(label_dir.glob("*_label.nii*"))
    
    print(f"找到 {len(label_files)} 个标签文件")
    print(f"将使用以下映射生成边缘: {chamber_to_edge_map}")
    print(f"边缘类型: {edge_type}")
    print(f"膨胀大小: {dilate_size}")
    print("=" * 60)
    
    for label_path in label_files:
        print(f"\n处理: {label_path.name}")
        
        label_img = sitk.ReadImage(str(label_path))
        label_array = sitk.GetArrayFromImage(label_img)
        
        # 初始化一个空的组合边缘图
        combined_edge_array = np.zeros_like(label_array, dtype=np.uint8)

        # 遍历每个心腔
        for seg_id, edge_id in chamber_to_edge_map.items():
            
            # 创建当前心腔的二值mask
            chamber_mask = (label_array == seg_id).astype(np.uint8)
            
            if np.sum(chamber_mask) == 0:
                continue

            print(f"  + 处理分割ID {seg_id}, 分配边缘ID {edge_id}...")

            # 生成单一边缘 (值为 0 或 1)
            single_binary_edge = np.zeros_like(label_array, dtype=np.uint8)
            
            if label_array.ndim == 3:
                for slice_idx in range(chamber_mask.shape[0]):
                    mask_slice = chamber_mask[slice_idx]
                    if np.all(mask_slice == 0):
                        continue
                    
                    edge_slice = generate_edge_from_segmentation(
                        mask_slice, edge_type=edge_type, dilate_size=dilate_size
                    )
                    single_binary_edge[slice_idx] = edge_slice
            else: # 2D
                single_binary_edge = generate_edge_from_segmentation(
                    chamber_mask, edge_type=edge_type, dilate_size=dilate_size
                )
            
            # 将二值边缘 (0/1) 乘以新的边缘ID
            # 然后使用 np.maximum 合并到主边缘图中
            combined_edge_array = np.maximum(combined_edge_array, (single_binary_edge * edge_id).astype(np.uint8))

        # 保存组合后的边缘标签
        edge_img = sitk.GetImageFromArray(combined_edge_array)
        edge_img.CopyInformation(label_img)
        
        # output_name = label_path.name.replace('.nii.gz', '_multi_edge.nii.gz')
        output_name = label_path.name.replace('.nii.gz', '_edge.nii.gz')
        output_path = output_dir / output_name
        
        sitk.WriteImage(edge_img, str(output_path), useCompression=True)
        print(f"  ✓ 组合边缘图保存到: {output_path}")

    print(f"\n{'='*60}")
    print(f"✓ 完成！共处理 {len(label_files)} 个文件")
    print(f"输出目录: {output_dir}")

def visualize_edge_overlay(
    image_path: str,
    label_path: str,
    edge_path: str,
    output_path: str
):
    """
    可视化：图像 + 分割 + 边缘叠加
    """
    import matplotlib.pyplot as plt
    
    # 读取
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    edge = sitk.GetArrayFromImage(sitk.ReadImage(edge_path))
    
    # 如果是3D，取中间切片
    if image.ndim == 3:
        mid_slice = image.shape[0] // 2
        image = image[mid_slice]
        label = label[mid_slice]
        edge = edge[mid_slice]
    
    # 归一化图像
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原始图像
    axes[0, 0].imshow(image_norm, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 分割标签
    axes[0, 1].imshow(image_norm, cmap='gray')
    axes[0, 1].imshow(label, alpha=0.5, cmap='jet')
    axes[0, 1].set_title('Segmentation')
    axes[0, 1].axis('off')
    
    # 边缘标签
    axes[1, 0].imshow(edge, cmap='gray')
    axes[1, 0].set_title('Edge Label')
    axes[1, 0].axis('off')
    
    # 叠加
    axes[1, 1].imshow(image_norm, cmap='gray')
    axes[1, 1].imshow(edge, alpha=0.7, cmap='Reds')
    axes[1, 1].set_title('Image + Edge Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 可视化保存到: {output_path}")


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数"""
    
    # 配置路径
    # image_dir = r"D:\dataset\TEECT_data\ct\ct_train_1004_image"
    # label_dir = r"D:\dataset\TEECT_data\tee\patient-1-4"
    label_dir = r"D:\dataset\TEECT_data\ct\ct_train_1004_label"
    output_dir = r"D:\dataset\TEECT_data\ct\ct_train_1004_edge"
    
    # chamber_to_edge_map = {
    #     420: 1,  # LV (左心房) 边缘将被标记为 1
    #     500: 2,  # LA (左心室) 边缘将被标记为 2
    #     # 根据需要添加其他腔室，并为它们分配唯一的边缘ID
    #     600: 4,  # RV (右心室)
    #     550: 3,  # RA (右心房)
    # }
    # chamber_to_edge_map = {
    #     2: 1,  # LV (左心房) 边缘将被标记为 1
    #     1: 2,  # LA (左心室) 边缘将被标记为 2
    #     # 根据需要添加其他腔室，并为它们分配唯一的边缘ID
    #     4: 4,  # RV (右心室)
    #     3: 3,  # RA (右心房)
    # }
    # chamber_to_edge_map = {
    #     2: 1,  # LV (左心房) 边缘将被标记为 1
    #     1: 1,  # LA (左心室) 边缘将被标记为 2
    #     # 根据需要添加其他腔室，并为它们分配唯一的边缘ID
    #     4: 1,  # RV (右心室)
    #     3: 1,  # RA (右心房)
    # }
    chamber_to_edge_map = {
        420: 1,  # LV (左心房) 边缘将被标记为 1
        500: 1,  # LA (左心室) 边缘将被标记为 2
        # 根据需要添加其他腔室，并为它们分配唯一的边缘ID
        600: 1,  # RV (右心室)
        550: 1,  # RA (右心房)
    }

    # 批量生成多标签边缘
    batch_generate_edge_labels(
        label_dir=label_dir,
        output_dir=output_dir,
        chamber_to_edge_map=chamber_to_edge_map,
        edge_type='contour',  # 'contour'最适合为独立对象提取轮廓
        dilate_size=1         # 边缘膨胀1个像素
    )
    
    # # # 可视化示例
    # visualize_edge_overlay(
    #     image_path=r"D:\dataset\TEECT_data\ct\ct_train_1004_image\slice_062_t0.0_rx0_ry0.nii.gz",
    #     label_path=r"D:\dataset\TEECT_data\ct\ct_train_1004_label\slice_062_t0.0_rx0_ry0_label.nii.gz",
    #     edge_path=r"D:\dataset\TEECT_data\edge_labels\slice_062_t0.0_rx0_ry0_edge.nii.gz",
    #     output_path=r"D:\dataset\TEECT_data\edge_labels\slice_062_t0.0_rx0_ry0_visualization.png"
    # )


if __name__ == "__main__":
    main()