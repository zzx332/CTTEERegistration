"""
使用示例 - 展示如何使用CT3DSliceGenerator
"""

from main import CT3DSliceGenerator
from config import SliceConfig
import numpy as np


def example_1_basic_usage():
    """
    示例1: 基本使用
    从轴向平面生成125个候选切片
    """
    print("\n" + "="*60)
    print("示例1: 基本使用")
    print("="*60)
    
    # 创建生成器
    generator = CT3DSliceGenerator(SliceConfig.CT_IMAGE_PATH)
    
    # 生成切片平面
    slice_planes = generator.generate_slice_planes(
        center_point=SliceConfig.CENTER_POINT,
        normal_vector=SliceConfig.NORMAL_VECTOR,
        translation_range=SliceConfig.TRANSLATION_RANGE,
        translation_step=SliceConfig.TRANSLATION_STEP,
        rotation_range=SliceConfig.ROTATION_RANGE,
        rotation_step=SliceConfig.ROTATION_STEP,
        slice_size=SliceConfig.SLICE_SIZE,
        slice_spacing=SliceConfig.SLICE_SPACING
    )
    
    # 提取所有切片
    slices = generator.extract_all_slices(
        slice_planes=slice_planes,
        output_dir=SliceConfig.OUTPUT_DIR if SliceConfig.SAVE_IMAGES else None
    )
    
    # 可视化
    if SliceConfig.VISUALIZE_SAMPLES:
        generator.visualize_sample_slices(
            slices, 
            slice_planes, 
            num_samples=SliceConfig.NUM_VISUALIZATION_SAMPLES
        )
    
    return slices, slice_planes


def example_2_custom_plane():
    """
    示例2: 自定义平面方向
    从冠状平面生成切片
    """
    print("\n" + "="*60)
    print("示例2: 自定义平面 - 冠状平面")
    print("="*60)
    
    generator = CT3DSliceGenerator(SliceConfig.CT_IMAGE_PATH)
    
    # 使用冠状平面
    slice_planes = generator.generate_slice_planes(
        center_point=(0, 0, 0),
        normal_vector=(0, 1, 0),  # 冠状平面
        translation_range=5.0,
        translation_step=2.5,
        rotation_range=90.0,
        rotation_step=45.0,
        slice_size=(256, 256),
        slice_spacing=(1.0, 1.0)
    )
    
    # 只提取部分切片（演示用）
    sample_indices = [0, len(slice_planes)//4, len(slice_planes)//2, 
                      3*len(slice_planes)//4, len(slice_planes)-1]
    
    slices = []
    for idx in sample_indices:
        slice_array = generator.extract_slice(slice_planes[idx])
        slices.append(slice_array)
    
    return slices, [slice_planes[i] for i in sample_indices]


def example_3_oblique_plane():
    """
    示例3: 斜切面
    使用任意角度的斜切面
    """
    print("\n" + "="*60)
    print("示例3: 斜切面")
    print("="*60)
    
    generator = CT3DSliceGenerator(SliceConfig.CT_IMAGE_PATH)
    
    # 使用斜切面（45度角）
    normal = np.array([1, 1, 1])
    normal = normal / np.linalg.norm(normal)
    
    slice_planes = generator.generate_slice_planes(
        center_point=(0, 0, 0),
        normal_vector=tuple(normal),
        translation_range=5.0,
        translation_step=2.5,
        rotation_range=90.0,
        rotation_step=45.0,
        slice_size=(256, 256),
        slice_spacing=(1.0, 1.0)
    )
    
    # 提取第一个切片作为示例
    slice_array = generator.extract_slice(slice_planes[0])
    
    return [slice_array], [slice_planes[0]]


def example_4_different_resolutions():
    """
    示例4: 不同分辨率
    生成不同大小和间距的切片
    """
    print("\n" + "="*60)
    print("示例4: 不同分辨率")
    print("="*60)
    
    generator = CT3DSliceGenerator(SliceConfig.CT_IMAGE_PATH)
    
    resolutions = [
        ((128, 128), (2.0, 2.0)),   # 低分辨率
        ((256, 256), (1.0, 1.0)),   # 中分辨率
        ((512, 512), (0.5, 0.5)),   # 高分辨率
    ]
    
    all_slices = []
    
    for size, spacing in resolutions:
        print(f"\n生成分辨率: {size}, 间距: {spacing}mm")
        
        # 只生成中心平面（不旋转不平移）
        slice_planes = generator.generate_slice_planes(
            center_point=(0, 0, 0),
            normal_vector=(0, 0, 1),
            translation_range=0.0,
            translation_step=2.5,
            rotation_range=0.0,
            rotation_step=45.0,
            slice_size=size,
            slice_spacing=spacing
        )
        
        slice_array = generator.extract_slice(slice_planes[0])
        all_slices.append(slice_array)
    
    return all_slices


def example_5_batch_processing():
    """
    示例5: 批量处理
    处理多个CT图像
    """
    print("\n" + "="*60)
    print("示例5: 批量处理")
    print("="*60)
    
    # 假设有多个CT图像
    ct_image_paths = [
        "data/patient_001/ct_image.nii.gz",
        "data/patient_002/ct_image.nii.gz",
        "data/patient_003/ct_image.nii.gz",
    ]
    
    for i, ct_path in enumerate(ct_image_paths):
        print(f"\n处理图像 {i+1}/{len(ct_image_paths)}: {ct_path}")
        
        try:
            generator = CT3DSliceGenerator(ct_path)
            
            slice_planes = generator.generate_slice_planes(
                center_point=(0, 0, 0),
                normal_vector=(0, 0, 1),
                translation_range=5.0,
                translation_step=2.5,
                rotation_range=90.0,
                rotation_step=45.0,
                slice_size=(256, 256),
                slice_spacing=(1.0, 1.0)
            )
            
            output_dir = f"output_slices/patient_{i+1:03d}"
            slices = generator.extract_all_slices(slice_planes, output_dir)
            
            print(f"完成！生成 {len(slices)} 个切片")
            
        except Exception as e:
            print(f"处理失败: {e}")
            continue


if __name__ == "__main__":
    # 运行示例1（基本使用）
    example_1_basic_usage()
    
    # 根据需要运行其他示例
    # example_2_custom_plane()
    # example_3_oblique_plane()
    # example_4_different_resolutions()
    # example_5_batch_processing()