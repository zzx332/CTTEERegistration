"""
配准测试脚本 - 用于验证配准功能
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tee_ct_registration import RigidRegistration2D
from registration_config import RegistrationPresets


def create_test_images():
    """
    创建测试图像（用于功能验证）
    """
    print("创建测试图像...")
    
    # 创建固定图像（256x256）
    fixed_array = np.zeros((256, 256), dtype=np.float32)
    # 添加一些特征：矩形
    fixed_array[80:180, 80:180] = 100
    fixed_array[100:160, 100:160] = 200
    
    # 创建移动图像（带有旋转和平移的固定图像）
    moving_array = np.zeros((256, 256), dtype=np.float32)
    moving_array[70:170, 90:190] = 100
    moving_array[90:150, 110:170] = 200
    
    # 转换为SimpleITK图像
    fixed_image = sitk.GetImageFromArray(fixed_array)
    fixed_image.SetSpacing([1.0, 1.0])
    fixed_image.SetOrigin([0.0, 0.0])
    
    moving_image = sitk.GetImageFromArray(moving_array)
    moving_image.SetSpacing([1.0, 1.0])
    moving_image.SetOrigin([0.0, 0.0])
    
    # 保存测试图像
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    fixed_path = test_dir / "test_fixed.nii.gz"
    moving_path = test_dir / "test_moving.nii.gz"
    
    sitk.WriteImage(fixed_image, str(fixed_path))
    sitk.WriteImage(moving_image, str(moving_path))
    
    print(f"  固定图像: {fixed_path}")
    print(f"  移动图像: {moving_path}")
    
    return str(fixed_path), str(moving_path)


def test_basic_registration():
    """
    测试1: 基本配准功能
    """
    print("\n" + "=" * 60)
    print("测试1: 基本配准功能")
    print("=" * 60)
    
    # 创建测试图像
    fixed_path, moving_path = create_test_images()
    
    # 创建配准器
    registrator = RigidRegistration2D()
    
    # 执行配准（使用快速模式）
    try:
        registered_image, transform = registrator.register(
            fixed_image_path=fixed_path,
            moving_image_path=moving_path,
            output_dir="test_results/basic",
            **RegistrationPresets.FAST
        )
        
        print("\n✓ 基本配准测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 基本配准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preset_configs():
    """
    测试2: 预定义配置
    """
    print("\n" + "=" * 60)
    print("测试2: 预定义配置")
    print("=" * 60)
    
    fixed_path, moving_path = create_test_images()
    registrator = RigidRegistration2D()
    
    presets = {
        'FAST': RegistrationPresets.FAST,
        'STANDARD': RegistrationPresets.STANDARD,
    }
    
    for name, config in presets.items():
        print(f"\n测试配置: {name}")
        try:
            registered_image, transform = registrator.register(
                fixed_image_path=fixed_path,
                moving_image_path=moving_path,
                output_dir=f"test_results/{name.lower()}",
                save_visualization=False,  # 不保存可视化以加快速度
                **config
            )
            print(f"  ✓ {name} 配置测试通过")
            
        except Exception as e:
            print(f"  ✗ {name} 配置测试失败: {e}")
            return False
    
    print("\n✓ 预定义配置测试通过")
    return True


def test_transform_save_load():
    """
    测试3: 变换保存和加载
    """
    print("\n" + "=" * 60)
    print("测试3: 变换保存和加载")
    print("=" * 60)
    
    fixed_path, moving_path = create_test_images()
    registrator = RigidRegistration2D()
    
    try:
        # 执行配准并保存变换
        print("\n执行配准...")
        registered_image, transform = registrator.register(
            fixed_image_path=fixed_path,
            moving_image_path=moving_path,
            output_dir="test_results/transform_test",
            **RegistrationPresets.FAST
        )
        
        # 加载变换
        print("\n加载变换文件...")
        transform_path = "test_results/transform_test/transform.tfm"
        loaded_transform = sitk.ReadTransform(transform_path)
        print(f"  变换已加载: {transform_path}")
        
        # 应用变换到新图像
        print("\n应用变换到新图像...")
        output_path = "test_results/transform_test/applied_transform.nii.gz"
        registrator.apply_transform_to_file(
            moving_image_path=moving_path,
            fixed_image_path=fixed_path,
            transform=loaded_transform,
            output_path=output_path
        )
        
        print("\n✓ 变换保存和加载测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 变换保存和加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_data():
    """
    测试4: 真实数据配准（如果存在）
    """
    print("\n" + "=" * 60)
    print("测试4: 真实数据配准")
    print("=" * 60)
    
    # 检查是否存在真实数据
    fixed_path = r"D:\dataset\TEECT_data\ct\patient-1-4\slice_123_image.nii.gz"
    moving_path = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_010_image.nii.gz"
    
    if not Path(fixed_path).exists() or not Path(moving_path).exists():
        print("真实数据不存在，跳过此测试")
        return True
    
    print("\n找到真实数据，开始配准...")
    registrator = RigidRegistration2D()
    
    try:
        registered_image, transform = registrator.register(
            fixed_image_path=fixed_path,
            moving_image_path=moving_path,
            output_dir="test_results/real_data",
            **RegistrationPresets.STANDARD
        )
        
        print("\n✓ 真实数据配准测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 真实数据配准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """
    运行所有测试
    """
    print("=" * 60)
    print("CT-TEE 配准功能测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    tests = [
        ("基本配准", test_basic_registration),
        ("预定义配置", test_preset_configs),
        ("变换保存加载", test_transform_save_load),
        ("真实数据配准", test_real_data),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n测试 '{test_name}' 发生异常: {e}")
            results.append((test_name, False))
    
    # 打印测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

