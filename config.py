"""
配置文件 - 用于设置CT切片生成的参数
"""

class SliceConfig:
    """切片生成配置"""
    
    # ===== 输入配置 =====
    # CT图像路径（支持NIfTI格式或DICOM文件夹）
    CT_IMAGE_PATH = "data/ct_image.nii.gz"
    
    # ===== 初始平面配置 =====
    # 初始平面中心点（物理坐标，单位：mm）
    # 可以通过ITK-SNAP等工具获取感兴趣区域的坐标
    CENTER_POINT = (0.0, 0.0, 0.0)
    
    # 初始平面法向量
    # (0, 0, 1) - 轴向平面 (Axial)
    # (0, 1, 0) - 冠状平面 (Coronal)
    # (1, 0, 0) - 矢状平面 (Sagittal)
    NORMAL_VECTOR = (0.0, 0.0, 1.0)
    
    # ===== 平移参数 =====
    TRANSLATION_RANGE = 5.0   # ±5mm
    TRANSLATION_STEP = 2.5    # 2.5mm步长（生成5个位置）
    
    # ===== 旋转参数 =====
    ROTATION_RANGE = 90.0     # ±90度
    ROTATION_STEP = 45.0      # 45度步长（生成5个角度）
    
    # ===== 输出切片配置 =====
    SLICE_SIZE = (256, 256)         # 输出切片大小（像素）
    SLICE_SPACING = (1.0, 1.0)      # 切片内像素间距（mm）
    
    # ===== 输出配置 =====
    OUTPUT_DIR = "output_slices"    # 输出目录
    SAVE_IMAGES = True              # 是否保存切片图像
    VISUALIZE_SAMPLES = True        # 是否生成样本可视化
    NUM_VISUALIZATION_SAMPLES = 9   # 可视化样本数量
    
    # ===== HU窗口配置（用于显示和保存） =====
    WINDOW_CENTER = 40              # 窗位（软组织）
    WINDOW_WIDTH = 400              # 窗宽（软组织）
    
    @classmethod
    def get_window_range(cls):
        """获取窗口的HU值范围"""
        min_hu = cls.WINDOW_CENTER - cls.WINDOW_WIDTH / 2
        max_hu = cls.WINDOW_CENTER + cls.WINDOW_WIDTH / 2
        return min_hu, max_hu


# 预定义的窗口设置
class WindowPresets:
    """常用的CT窗口预设"""
    
    # 软组织窗
    SOFT_TISSUE = {'center': 40, 'width': 400}
    
    # 肺窗
    LUNG = {'center': -600, 'width': 1500}
    
    # 骨窗
    BONE = {'center': 400, 'width': 1800}
    
    # 脑窗
    BRAIN = {'center': 40, 'width': 80}
    
    # 肝窗
    LIVER = {'center': 30, 'width': 150}