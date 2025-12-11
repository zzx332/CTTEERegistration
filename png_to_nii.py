import SimpleITK as sitk
import numpy as np
from pathlib import Path
import cv2
from typing import Union, List, Optional


def png_to_nii(
    input_path: Union[str, Path, List[Union[str, Path]]],
    output_path: Union[str, Path],
    spacing: Optional[tuple] = None,
    origin: Optional[tuple] = None,
    is_grayscale: bool = True,
    sort_files: bool = True
):
    """
    将PNG图像转换为NII.GZ格式
    
    Args:
        input_path: 单个PNG文件路径，或包含PNG文件的目录路径，或PNG文件路径列表
        output_path: 输出NII.GZ文件路径
        spacing: 体素间距 (x, y, z) 或 (x, y)，默认(1.0, 1.0, 1.0)或(1.0, 1.0)
        origin: 图像原点坐标，默认(0.0, 0.0, 0.0)或(0.0, 0.0)
        is_grayscale: 是否以灰度模式读取（True=单通道，False=RGB三通道）
        sort_files: 如果是目录，是否对文件进行排序
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 确定输入文件列表
    if input_path.is_file():
        # 单个文件
        png_files = [input_path]
    elif input_path.is_dir():
        # 目录中的所有PNG文件
        png_files = sorted(input_path.glob("*.png")) if sort_files else list(input_path.glob("*.png"))
        if not png_files:
            raise ValueError(f"目录中没有找到PNG文件: {input_path}")
    else:
        raise ValueError(f"无效的输入路径: {input_path}")
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    # 读取第一个图像以获取尺寸信息
    first_img = cv2.imread(str(png_files[0]), cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR)
    if first_img is None:
        raise ValueError(f"无法读取图像: {png_files[0]}")
    
    height, width = first_img.shape[:2]
    num_slices = len(png_files)
    
    print(f"图像尺寸: {width} x {height}")
    print(f"切片数量: {num_slices}")
    
    # 确定数组形状
    if is_grayscale:
        if num_slices == 1:
            # 2D图像: (height, width)
            array_shape = (height, width)
        else:
            # 3D图像: (num_slices, height, width) - SimpleITK使用(z, y, x)顺序
            array_shape = (num_slices, height, width)
    else:
        if num_slices == 1:
            # 2D RGB图像: (height, width, 3)
            array_shape = (height, width, 3)
        else:
            # 3D RGB图像: (num_slices, height, width, 3)
            array_shape = (num_slices, height, width, 3)
    
    # 创建数组
    if is_grayscale:
        volume_array = np.zeros(array_shape, dtype=np.uint8)
    else:
        volume_array = np.zeros(array_shape, dtype=np.uint8)
    
    # 读取所有图像
    for i, png_file in enumerate(png_files):
        # img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR).astype(dtype=np.float32)
        img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR)
        if img is None:
            print(f"警告: 无法读取 {png_file}，跳过")
            continue
        
        if is_grayscale:
            if num_slices == 1:
                volume_array = img
            else:
                volume_array[i] = img
        else:
            # OpenCV读取BGR，转换为RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if num_slices == 1:
                volume_array = img_rgb
            else:
                volume_array[i] = img_rgb
    
    # 转换为SimpleITK图像
    sitk_image = sitk.GetImageFromArray(volume_array)
    
    # 设置spacing
    if spacing is None:
        if num_slices == 1:
            spacing = (1.0, 1.0)
        else:
            spacing = (1.0, 1.0, 1.0)
    sitk_image.SetSpacing(spacing)
    
    # 设置origin
    if origin is None:
        if num_slices == 1:
            origin = (0.0, 0.0)
        else:
            origin = (0.0, 0.0, 0.0)
    sitk_image.SetOrigin(origin)
    # return sitk_image
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为NII.GZ
    sitk.WriteImage(sitk_image, str(output_path), useCompression=True)
    print(f"✓ 已保存: {output_path}")
    print(f"  图像尺寸: {sitk_image.GetSize()}")
    print(f"  Spacing: {sitk_image.GetSpacing()}")
    print(f"  Origin: {sitk_image.GetOrigin()}")


def main():
    """示例用法"""
    # 示例1: 单个PNG文件转换为2D NII.GZ
    import os

    data_path = r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images"
    output_path = r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images_nii"
    for file in os.listdir(data_path):
        png_to_nii(
            input_path=os.path.join(data_path, file, "A4C.png"),
            output_path=os.path.join(output_path, f"{file}_A4C.nii.gz"),
            spacing=(1, 1),  # 2D图像只需要(x, y)间距
            origin=(0.0, 0.0)
        )
    # png_to_nii(
    #     input_path=r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images\Patient_0036\A2C.png",
    #     output_path=r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Multi-view_preprocess_images\Patient_0036\A2C.nii.gz",
    #     spacing=(1, 1),  # 2D图像只需要(x, y)间距
    #     origin=(0.0, 0.0)
    # )
    
    # 示例2: PNG序列转换为3D NII.GZ
    # png_to_nii(
    #     input_path=r"D:\dataset\images\png_sequence",  # 包含多个PNG的目录
    #     output_path=r"D:\dataset\images\volume.nii.gz",
    #     spacing=(0.5, 0.5, 1.0),  # 3D图像需要(x, y, z)间距
    #     origin=(0.0, 0.0, 0.0)
    # )
    
    # 示例3: 指定PNG文件列表
    # png_files = [
    #     r"D:\dataset\images\slice_001.png",
    #     r"D:\dataset\images\slice_002.png",
    #     r"D:\dataset\images\slice_003.png",
    # ]
    # png_to_nii(
    #     input_path=png_files,
    #     output_path=r"D:\dataset\images\volume.nii.gz",
    #     spacing=(0.5, 0.5, 1.0)
    # )
    
    pass


if __name__ == "__main__":
    main()
