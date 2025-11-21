"""
重新映射CT分割标签
将四腔心标签映射到新的值，其他标签置零
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
import os


def remap_labels(
    input_path: str,
    output_path: str,
    label_mapping: dict
):
    """
    重新映射分割标签
    
    Args:
        input_path: 输入分割图路径
        output_path: 输出分割图路径
        label_mapping: 标签映射字典，如 {420: 1, 500: 2, ...}
    """
    # 读取分割图
    seg_img = sitk.ReadImage(input_path)
    seg_array = sitk.GetArrayFromImage(seg_img)
    
    print(f"处理: {Path(input_path).name}")
    print(f"  原始标签: {np.unique(seg_array)}")
    
    # 创建新的数组（初始化为0）
    new_array = np.zeros_like(seg_array, dtype=np.uint8)
    
    # 应用映射
    for old_label, new_label in label_mapping.items():
        mask = (seg_array == old_label)
        new_array[mask] = new_label
        pixel_count = np.sum(mask)
        if pixel_count > 0:
            print(f"  {old_label} -> {new_label}: {pixel_count} 像素")
    
    print(f"  映射后标签: {np.unique(new_array)}")
    
    # 转换回SimpleITK图像
    new_img = sitk.GetImageFromArray(new_array)
    new_img.CopyInformation(seg_img)
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sitk.WriteImage(new_img, str(output_path), useCompression=True)
    print(f"  ✓ 已保存: {output_path}\n")


def batch_remap_labels(
    input_dir: str,
    output_dir: str,
    label_mapping: dict,
    pattern: str = "*.nii*"
):
    """
    批量重新映射标签
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        label_mapping: 标签映射字典
        pattern: 文件匹配模式
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 查找所有标签文件
    label_files = sorted(input_dir.glob(pattern))
    
    print(f"找到 {len(label_files)} 个标签文件")
    print(f"标签映射: {label_mapping}")
    print("=" * 60 + "\n")
    
    for label_file in label_files:
        # 构造输出路径
        output_path = output_dir / label_file.name
        
        # 重新映射
        remap_labels(str(label_file), str(output_path), label_mapping)
    
    print("=" * 60)
    print(f"✓ 完成！共处理 {len(label_files)} 个文件")
    print(f"输出目录: {output_dir}")


def visualize_remapped_labels(
    original_path: str,
    remapped_path: str,
    output_path: str = "label_comparison.png"
):
    """
    可视化映射前后的对比
    
    Args:
        original_path: 原始标签路径
        remapped_path: 映射后标签路径
        output_path: 可视化输出路径
    """
    import matplotlib.pyplot as plt
    
    # 读取图像
    original_img = sitk.ReadImage(original_path)
    remapped_img = sitk.ReadImage(remapped_path)
    
    original_array = sitk.GetArrayFromImage(original_img).squeeze()
    remapped_array = sitk.GetArrayFromImage(remapped_img).squeeze()
    
    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原始标签
    im1 = axes[0].imshow(original_array, cmap='tab10', vmin=0, vmax=10)
    axes[0].set_title(f'Original Labels\n{np.unique(original_array)}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # 映射后标签
    im2 = axes[1].imshow(remapped_array, cmap='tab10', vmin=0, vmax=10)
    axes[1].set_title(f'Remapped Labels\n{np.unique(remapped_array)}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 可视化已保存: {output_path}")


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数"""
    
    # 定义标签映射
    chamber_to_edge_map = {
        420: 2,  # LV (左心房) -> 1
        500: 1,  # LA (左心室) -> 2
        600: 4,  # RV (右心室) -> 4
        550: 3,  # RA (右心房) -> 3
    }
    
    # 方法1：单个文件处理
    single_file_example = False
    if single_file_example:
        input_file = r"D:\dataset\TEECT_data\ct\ct_train_1004_label\slice_062_t0.0_rx0_ry0_label.nii.gz"
        output_file = r"D:\dataset\TEECT_data\ct\ct_train_1004_label_remapped\slice_062_t0.0_rx0_ry0_label.nii.gz"
        
        remap_labels(input_file, output_file, chamber_to_edge_map)
        
        # 可视化对比
        visualize_remapped_labels(
            input_file, 
            output_file,
            "label_remap_comparison.png"
        )
    
    # 方法2：批量处理
    batch_processing_example = True
    if batch_processing_example:
        input_dir = r"D:\dataset\TEECT_data\ct\ct_train_1004_label"
        output_dir = r"D:\dataset\TEECT_data\ct\ct_train_1004_label_remapped"
        
        batch_remap_labels(
            input_dir=input_dir,
            output_dir=output_dir,
            label_mapping=chamber_to_edge_map,
            pattern="*.nii*"  # 匹配所有标签文件
        )
        
        # 可视化第一个文件作为示例
        first_file = sorted(Path(input_dir).glob("*.nii*"))[0]
        output_file = Path(output_dir) / first_file.name
        
        if output_file.exists():
            visualize_remapped_labels(
                str(first_file),
                str(output_file),
                str(Path(output_dir) / "label_remap_comparison.png")
            )


if __name__ == "__main__":
    main()