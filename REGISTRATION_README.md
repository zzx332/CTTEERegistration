# CT-TEE 2D图像刚性配准

基于互信息的多模态2D医学图像刚性配准工具。

## 功能特性

- ✅ **多模态配准**：专为CT和TEE超声图像设计
- ✅ **互信息度量**：适合不同模态间的配准
- ✅ **刚性变换**：平移（2自由度）+ 旋转（1自由度）
- ✅ **多分辨率优化**：由粗到精的配准策略
- ✅ **自动初始化**：基于质心对齐
- ✅ **可视化结果**：自动生成配准前后对比图
- ✅ **收敛监控**：实时显示度量值变化

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- SimpleITK >= 2.1.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0

## 快速开始

### 1. 基本使用

```python
from tee_ct_registration import RigidRegistration2D

# 创建配准器
registrator = RigidRegistration2D()

# 执行配准
registered_image, transform = registrator.register(
    fixed_image_path="ct_slice.nii.gz",      # CT图像（固定）
    moving_image_path="tee_slice.nii.gz",    # TEE图像（移动）
    output_dir="results/",                    # 输出目录
    number_of_iterations=500,                 # 迭代次数
    sampling_percentage=0.2                   # 采样率
)
```

### 2. 使用预定义配置

```python
from tee_ct_registration import RigidRegistration2D
from registration_config import RegistrationPresets

registrator = RigidRegistration2D()

# 使用快速配准
registered_image, transform = registrator.register(
    fixed_image_path="ct_slice.nii.gz",
    moving_image_path="tee_slice.nii.gz",
    output_dir="results/",
    **RegistrationPresets.FAST
)
```

### 3. 应用已有变换

```python
# 将配准得到的变换应用到其他图像
registrator.apply_transform_to_file(
    moving_image_path="new_tee_slice.nii.gz",
    fixed_image_path="ct_slice.nii.gz",
    transform=transform,
    output_path="new_registered.nii.gz"
)
```

## 配置参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `number_of_iterations` | 500 | 最大迭代次数 |
| `learning_rate` | 1.0 | 学习率（控制每步更新幅度） |
| `min_step` | 0.001 | 最小步长（收敛判据） |
| `relaxation_factor` | 0.5 | 松弛因子（0.5-0.8） |
| `sampling_percentage` | 0.2 | 采样百分比（0-1） |

### 相似性度量

- **互信息（Mattes Mutual Information）**：适合多模态配准
- **直方图bins**：默认50个bins
- **采样策略**：随机采样

### 多分辨率策略

```python
shrink_factors = [4, 2, 1]      # 缩小因子
smoothing_sigmas = [2, 1, 0]    # 平滑程度
```

- 第1层：图像缩小4倍，sigma=2
- 第2层：图像缩小2倍，sigma=1
- 第3层：原始分辨率，sigma=0

## 输出文件

配准完成后，在输出目录生成以下文件：

```
output_dir/
├── registered_image.nii.gz      # 配准后的图像
├── transform.tfm                # 变换文件（可重用）
├── registration_result.png      # 可视化对比图
└── metric_curve.png            # 收敛曲线图
```

### 可视化说明

`registration_result.png` 包含6个子图：

**第一行（单独显示）：**
1. 固定图像（CT）
2. 移动图像配准前（TEE）
3. 移动图像配准后（TEE）

**第二行（叠加显示）：**
4. 配准前叠加（红=CT，绿=TEE）
5. 配准后叠加（红=CT，绿=TEE）
6. 差异图（配准后的绝对差）

## 预定义配置方案

### FAST - 快速配准
```python
RegistrationPresets.FAST
```
- 迭代次数: 100
- 采样率: 10%
- 适用场景: 初步测试，快速验证

### STANDARD - 标准配准
```python
RegistrationPresets.STANDARD
```
- 迭代次数: 300
- 采样率: 20%
- 适用场景: 一般配准任务

### FINE - 精细配准
```python
RegistrationPresets.FINE
```
- 迭代次数: 1000
- 采样率: 30%
- 学习率: 0.5（更保守）
- 适用场景: 高精度要求

### ROBUST - 鲁棒配准
```python
RegistrationPresets.ROBUST
```
- 迭代次数: 500
- 学习率: 2.0（更激进）
- 适用场景: 初始位置差异大

## 批量配准

```python
from tee_ct_registration import RigidRegistration2D

registrator = RigidRegistration2D()

# 定义配准对列表
pairs = [
    {
        'fixed': 'ct_slice_001.nii.gz',
        'moving': 'tee_slice_001.nii.gz',
        'output': 'results/pair_001/'
    },
    {
        'fixed': 'ct_slice_002.nii.gz',
        'moving': 'tee_slice_002.nii.gz',
        'output': 'results/pair_002/'
    },
]

# 批量处理
for i, pair in enumerate(pairs):
    print(f"\n处理配准对 {i+1}/{len(pairs)}")
    try:
        registered, transform = registrator.register(
            fixed_image_path=pair['fixed'],
            moving_image_path=pair['moving'],
            output_dir=pair['output'],
            number_of_iterations=300
        )
    except Exception as e:
        print(f"配准失败: {e}")
        continue
```

## 常见问题

### Q1: 配准结果不理想？

**可能原因和解决方案：**

1. **初始位置差异太大**
   - 使用 `ROBUST` 配置
   - 增加学习率（如 2.0）
   - 考虑手动设置初始变换

2. **收敛到局部最优**
   - 增加迭代次数
   - 提高采样率（如 0.3）
   - 调整多分辨率层级

3. **图像对比度差异大**
   - 互信息已经适合多模态，但可以尝试预处理
   - 考虑直方图均衡化

### Q2: 如何设置初始变换？

```python
import SimpleITK as sitk

# 手动设置初始变换
initial_transform = sitk.Euler2DTransform()
initial_transform.SetTranslation([10.0, 5.0])  # 初始平移
initial_transform.SetAngle(0.1)                # 初始旋转（弧度）

registered_image, transform = registrator.register(
    ...,
    initial_transform=initial_transform
)
```

### Q3: 如何加载已保存的变换？

```python
import SimpleITK as sitk

# 加载变换文件
transform = sitk.ReadTransform("transform.tfm")

# 应用到新图像
registrator.apply_transform_to_file(
    moving_image_path="new_image.nii.gz",
    fixed_image_path="reference.nii.gz",
    transform=transform,
    output_path="result.nii.gz"
)
```

### Q4: 配准速度太慢？

**优化建议：**

1. 减少采样率（如 0.1）
2. 减少迭代次数
3. 使用 `FAST` 预设
4. 减少多分辨率层级

### Q5: 如何评估配准质量？

**评估方法：**

1. **视觉检查**：查看叠加图（绿色和红色应重叠）
2. **度量值**：互信息值越大越好（注意是负值）
3. **差异图**：配准后差异应该减小
4. **解剖标志**：检查关键解剖结构是否对齐

## 变换参数说明

2D刚性变换有3个自由度：

```
变换参数:
  类型: 2D刚性变换 (Euler2D)
  旋转中心: (128.0, 128.0)      # 图像中心
  旋转角度: 15.30°              # 绕中心旋转
  平移: (5.23, -3.45) mm        # x和y方向平移
```

## 技术细节

### 优化算法

使用 **Regular Step Gradient Descent**：
- 基于梯度的优化
- 自适应步长（通过relaxation factor）
- 支持多分辨率策略

### 插值方法

- **配准过程**: 线性插值（速度快）
- **最终重采样**: 线性插值（可改为B样条获得更好质量）

### 坐标系统

- SimpleITK使用物理坐标系统（mm）
- 自动处理spacing、origin和direction
- 输出变换在物理空间定义

## 进阶用法

### 自定义度量函数

```python
registration = sitk.ImageRegistrationMethod()

# 可选的度量函数
# 1. 互信息（当前使用）
registration.SetMetricAsMattesMutualInformation(50)

# 2. 归一化互信息
# registration.SetMetricAsJointHistogramMutualInformation(50)

# 3. 均方差（适合单模态）
# registration.SetMetricAsMeanSquares()
```

### 自定义优化器

```python
# 使用不同的优化器
# 1. LBFGSB（更快但需要更多内存）
registration.SetOptimizerAsLBFGSB()

# 2. Powell（无梯度优化）
registration.SetOptimizerAsPowell()
```

## 参考文献

1. Mattes, D., et al. "PET-CT image registration in the chest using free-form deformations." IEEE TMI, 2003.
2. Klein, S., et al. "elastix: a toolbox for intensity-based medical image registration." IEEE TMI, 2010.

## 许可证

MIT License

## 作者

AI Assistant

## 更新日志

- v1.0 (2025-01-12): 初始版本，支持基于互信息的2D刚性配准

