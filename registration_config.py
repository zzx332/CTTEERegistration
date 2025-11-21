"""
CT-TEE 2D刚性配准配置文件
"""

class RegistrationConfig:
    """配准参数配置"""
    
    # ===== 输入输出路径 =====
    FIXED_IMAGE = r"D:\dataset\TEECT_data\ct\patient-1-4\slice_123_image.nii.gz"
    MOVING_IMAGE = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_010_image.nii.gz"
    OUTPUT_DIR = r"D:\dataset\TEECT_data\registration_results\patient-1-4"
    
    # ===== 优化器参数 =====
    NUMBER_OF_ITERATIONS = 500        # 最大迭代次数
    LEARNING_RATE = 1.0              # 学习率
    MIN_STEP = 0.001                 # 最小步长（收敛条件）
    RELAXATION_FACTOR = 0.5          # 松弛因子（0.5-0.8之间）
    
    # ===== 相似性度量参数 =====
    NUMBER_OF_HISTOGRAM_BINS = 50    # 互信息直方图bins数量
    SAMPLING_PERCENTAGE = 0.2        # 采样百分比（0-1）
    
    # ===== 多分辨率策略 =====
    SHRINK_FACTORS = [4, 2, 1]       # 图像缩小因子（由粗到精）
    SMOOTHING_SIGMAS = [2, 1, 0]     # 平滑sigma值
    
    # ===== 输出控制 =====
    SAVE_VISUALIZATION = True        # 保存可视化结果
    SAVE_TRANSFORM = True            # 保存变换文件
    VERBOSE = True                   # 详细输出
    
    # ===== 预处理参数 =====
    # 可选：图像预处理
    NORMALIZE_INTENSITY = False      # 是否归一化强度
    HISTOGRAM_MATCHING = False       # 是否使用直方图匹配
    
    @classmethod
    def get_config_dict(cls):
        """获取配置字典"""
        return {
            'number_of_iterations': cls.NUMBER_OF_ITERATIONS,
            'learning_rate': cls.LEARNING_RATE,
            'min_step': cls.MIN_STEP,
            'relaxation_factor': cls.RELAXATION_FACTOR,
            'sampling_percentage': cls.SAMPLING_PERCENTAGE,
            'save_visualization': cls.SAVE_VISUALIZATION,
        }


# 预定义的配置方案
class RegistrationPresets:
    """预定义的配准参数方案"""
    
    # 快速配准（适合初步测试）
    FAST = {
        'number_of_iterations': 100,
        'learning_rate': 1.0,
        'min_step': 0.01,
        'relaxation_factor': 0.5,
        'sampling_percentage': 0.1,
    }
    
    # 标准配准（平衡速度和精度）
    STANDARD = {
        'number_of_iterations': 300,
        'learning_rate': 1.0,
        'min_step': 0.001,
        'relaxation_factor': 0.5,
        'sampling_percentage': 0.2,
    }
    
    # 精细配准（高精度）
    FINE = {
        'number_of_iterations': 1000,
        'learning_rate': 0.5,
        'min_step': 0.0001,
        'relaxation_factor': 0.6,
        'sampling_percentage': 0.3,
    }
    
    # 鲁棒配准（处理大位移）
    ROBUST = {
        'number_of_iterations': 500,
        'learning_rate': 2.0,
        'min_step': 0.001,
        'relaxation_factor': 0.4,
        'sampling_percentage': 0.25,
    }

