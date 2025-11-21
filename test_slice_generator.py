
import unittest
import numpy as np
from main import CT3DSliceGenerator
import tempfile
import SimpleITK as sitk


class TestCT3DSliceGenerator(unittest.TestCase):
    """测试CT3DSliceGenerator类"""
    
    @classmethod
    def setUpClass(cls):
        """创建测试用的CT图像"""
        # 创建一个简单的测试图像（100x100x100）
        image_array = np.random.randint(0, 100, (100, 100, 100), dtype=np.int16)
        cls.test_image = sitk.GetImageFromArray(image_array)
        cls.test_image.SetSpacing((1.0, 1.0, 1.0))
        cls.test_image.SetOrigin((0.0, 0.0, 0.0))
        
        # 保存到临时文件
        cls.temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        sitk.WriteImage(cls.test_image, cls.temp_file.name)
    
    def setUp(self):
        """每个测试前的设置"""
        self.generator = CT3DSliceGenerator(self.temp_file.name)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.generator.image)
        self.assertEqual(len(self.generator.spacing), 3)
        self.assertEqual(len(self.generator.origin), 3)
        self.assertEqual(len(self.generator.size), 3)
    
    def test_rotation_matrix(self):
        """测试旋转矩阵生成"""
        axis = np.array([0, 0, 1])
        angle = 90.0
        R = self.generator._create_rotation_matrix(axis, angle)
        
        # 验证旋转矩阵性质
        # 1. 行列式应为1
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=5)
        
        # 2. R * R^T = I
        identity = np.dot(R, R.T)
        np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=5)
    
    def test_generate_slice_planes(self):
        """测试切片平面生成"""
        slice_planes = self.generator.generate_slice_planes(
            center_point=(50, 50, 50),
            normal_vector=(0, 0, 1),
            translation_range=5.0,
            translation_step=2.5,
            rotation_range=90.0,
            rotation_step=45.0,
            slice_size=(128, 128),
            slice_spacing=(1.0, 1.0)
        )
        
        # 验证生成的平面数量
        expected_count = 5 * 5 * 5  # 125
        self.assertEqual(len(slice_planes), expected_count)
        
        # 验证每个平面的参数
        for plane in slice_planes:
            self.assertIn('center', plane)
            self.assertIn('x_axis', plane)
            self.assertIn('y_axis', plane)
            self.assertIn('normal', plane)
            self.assertIn('translation', plane)
            self.assertIn('rotation_x', plane)
            self.assertIn('rotation_y', plane)
            
            # 验证轴的正交性
            x_axis = plane['x_axis']
            y_axis = plane['y_axis']
            normal = plane['normal']
            
            # 检查单位向量
            self.assertAlmostEqual(np.linalg.norm(x_axis), 1.0, places=5)
            self.assertAlmostEqual(np.linalg.norm(y_axis), 1.0, places=5)
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=5)
            
            # 检查正交性
            self.assertAlmostEqual(np.dot(x_axis, y_axis), 0.0, places=5)
            self.assertAlmostEqual(np.dot(y_axis, normal), 0.0, places=5)
            self.assertAlmostEqual(np.dot(normal, x_axis), 0.0, places=5)
    
    def test_extract_slice(self):
        """测试切片提取"""
        plane_params = {
            'center': np.array([50.0, 50.0, 50.0]),
            'x_axis': np.array([1.0, 0.0, 0.0]),
            'y_axis': np.array([0.0, 1.0, 0.0]),
            'normal': np.array([0.0, 0.0, 1.0]),
            'size': (64, 64),
            'spacing': (1.0, 1.0),
            'translation': 0.0,
            'rotation_x': 0.0,
            'rotation_y': 0.0,
            'id': 0
        }
        
        slice_array = self.generator.extract_slice(plane_params)
        
        # 验证切片尺寸
        self.assertEqual(slice_array.shape, (64, 64))
        
        # 验证数据类型
        self.assertTrue(np.issubdtype(slice_array.dtype, np.number))
    
    def test_translation_range(self):
        """测试平移范围"""
        slice_planes = self.generator.generate_slice_planes(
            center_point=(50, 50, 50),
            normal_vector=(0, 0, 1),
            translation_range=5.0,
            translation_step=2.5,
            rotation_range=0.0,
            rotation_step=45.0
        )
        
        # 提取所有平移值
        translations = set(plane['translation'] for plane in slice_planes)
        expected_translations = {-5.0, -2.5, 0.0, 2.5, 5.0}
        
        self.assertEqual(translations, expected_translations)
    
    def test_rotation_range(self):
        """测试旋转范围"""
        slice_planes = self.generator.generate_slice_planes(
            center_point=(50, 50, 50),
            normal_vector=(0, 0, 1),
            translation_range=0.0,
            translation_step=2.5,
            rotation_range=90.0,
            rotation_step=45.0
        )
        
        # 提取所有旋转值
        rotations_x = set(plane['rotation_x'] for plane in slice_planes)
        rotations_y = set(plane['rotation_y'] for plane in slice_planes)
        expected_rotations = {-90.0, -45.0, 0.0, 45.0, 90.0}
        
        self.assertEqual(rotations_x, expected_rotations)
        self.assertEqual(rotations_y, expected_rotations)
    
    def test_different_normal_vectors(self):
        """测试不同的法向量"""
        normals = [
            (0, 0, 1),   # 轴向
            (0, 1, 0),   # 冠状
            (1, 0, 0),   # 矢状
            (1, 1, 1),   # 斜面
        ]
        
        for normal in normals:
            slice_planes = self.generator.generate_slice_planes(
                center_point=(50, 50, 50),
                normal_vector=normal,
                translation_range=0.0,
                translation_step=2.5,
                rotation_range=0.0,
                rotation_step=45.0
            )
            
            self.assertEqual(len(slice_planes), 1)
            
            # 验证法向量被正确归一化
            normalized_normal = slice_planes[0]['normal']
            expected_normal = np.array(normal) / np.linalg.norm(normal)
            np.testing.assert_array_almost_equal(
                normalized_normal, 
                expected_normal, 
                decimal=5
            )


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def test_slice_generation_speed(self):
        """测试切片生成速度"""
        import time
        
        # 创建测试图像
        image_array = np.random.randint(0, 100, (200, 200, 200), dtype=np.int16)
        test_image = sitk.GetImageFromArray(image_array)
        test_image.SetSpacing((1.0, 1.0, 1.0))
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        sitk.WriteImage(test_image, temp_file.name)
        
        generator = CT3DSliceGenerator(temp_file.name)
        
        # 测量生成平面参数的时间
        start_time = time.time()
        slice_planes = generator.generate_slice_planes(
            center_point=(100, 100, 100),
            normal_vector=(0, 0, 1)
        )
        param_time = time.time() - start_time
        
        print(f"\n生成125个平面参数耗时: {param_time:.3f}秒")
        
        # 测量提取单个切片的时间
        start_time = time.time()
        slice_array = generator.extract_slice(slice_planes[0])
        slice_time = time.time() - start_time
        
        print(f"提取单个切片耗时: {slice_time:.3f}秒")
        print(f"预计提取所有125个切片耗时: {slice_time * 125:.1f}秒")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestCT3DSliceGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    result = run_tests()
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结:")
    print(f"  运行测试: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print("="*60)