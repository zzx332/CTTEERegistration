# 多标签ICP配准方法说明

## 概述

多标签ICP配准方法允许你将四腔心的不同心腔（左心室、左心房、右心室、右心房）分别对齐，避免不同心腔之间的错误匹配，从而提高配准精度。

## 核心改进

### 传统ICP的问题
- **混淆匹配**: 所有边缘点混在一起，左心室的点可能错误地匹配到右心室
- **精度受限**: 不同心腔的形状差异导致配准误差

### 多标签ICP的优势
1. **语义约束**: 左心室只匹配左心室，避免跨心腔的错误匹配
2. **更高精度**: 每个心腔独立优化，配准更准确
3. **可调权重**: 可以对重要的心腔（如左心室）增加权重
4. **灵活选择**: 可以只使用部分标签进行配准

## 标签定义

```python
CHAMBER_LABELS = {
    1: 'LV',  # 左心室 Left Ventricle
    2: 'LA',  # 左心房 Left Atrium
    3: 'RV',  # 右心室 Right Ventricle
    4: 'RA'   # 右心房 Right Atrium
}
```

## 使用方法

### 基本用法

```python
from 2D_3D_registration_boundary_gps import TwoD_ThreeD_Registration

# 初始化
registrator = TwoD_ThreeD_Registration()

# 加载图像和标签mask
registrator.load_images(ct_path, ultrasound_path, ct_mask_path, us_mask_path)

# 执行多标签ICP配准
best_params, result_dict = registrator.register_icp_multi_label(
    initial_params=initial_params,
    max_iterations=20,
    tolerance=0.1,
    max_correspondence_dist=30.0,
    inner_opt_iterations=30,
    min_iterations=5,
    use_gps=False,
    labels=[1, 2, 3, 4],  # 使用所有四腔
    label_weights=None  # 等权重
)

# 可视化结果
registrator.visualize_result(result_dict, output_dir)
registrator.visualize_multi_label_correspondences(result_dict, output_dir)
```

### 高级用法：自定义权重

```python
# 对左心室和左心房赋予更高权重
best_params, result_dict = registrator.register_icp_multi_label(
    initial_params=initial_params,
    labels=[1, 2, 3, 4],
    label_weights={
        1: 2.0,  # LV - 左心室（权重加倍）
        2: 2.0,  # LA - 左心房（权重加倍）
        3: 1.0,  # RV - 右心室
        4: 1.0   # RA - 右心房
    }
)
```

### 高级用法：仅使用部分标签

```python
# 只使用左心室和左心房进行配准
best_params, result_dict = registrator.register_icp_multi_label(
    initial_params=initial_params,
    labels=[1, 2],  # 仅LV和LA
    label_weights=None
)
```

## 主要参数说明

### register_icp_multi_label 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `initial_params` | np.ndarray | None | 初始8参数 [α, β, γ, tx, ty, tz, sx, sy] |
| `max_iterations` | int | 50 | 最大ICP外层迭代次数 |
| `tolerance` | float | 1e-4 | 收敛阈值（平均距离变化，mm） |
| `max_correspondence_dist` | float | 50.0 | 最大对应点距离（mm） |
| `inner_opt_iterations` | int | 50 | 每次ICP迭代内部的优化迭代次数 |
| `min_iterations` | int | 10 | 最小迭代次数，避免过早停止 |
| `use_gps` | bool | True | 是否使用GPS优化（False则使用Powell） |
| `labels` | List[int] | [1,2,3,4] | 要使用的标签列表 |
| `label_weights` | Dict[int,float] | None | 各标签的权重（None表示等权） |

## 新增方法

### 1. extract_edge_points_by_label
按标签分别提取边缘点

```python
label_points = registrator.extract_edge_points_by_label(
    image=image,
    mask=mask,
    labels=[1, 2, 3, 4],
    subsample=3
)
# 返回: {label_id: edge_points (N, 2)} 字典
```

### 2. compute_multi_label_correspondence_cost
计算多标签对应代价

```python
cost = registrator.compute_multi_label_correspondence_cost(
    ct_label_points=ct_label_points,
    us_label_points=us_label_points,
    max_correspondence_dist=30.0,
    label_weights={1: 2.0, 2: 2.0, 3: 1.0, 4: 1.0}
)
```

### 3. visualize_multi_label_correspondences
可视化多标签配准结果

```python
registrator.visualize_multi_label_correspondences(
    result_dict=result_dict,
    output_dir=output_dir,
    max_correspondence_dist=30.0
)
```

## 输出结果

### 可视化输出
1. `registration_result.png` - 整体配准结果（4张子图）
2. `multi_label_registration.png` - 分标签配准结果（2x2布局）
3. `correspondence_visualization.png` - 对应点可视化（仅在使用普通ICP时）

### 统计报告
1. `multi_label_stats.txt` - 详细的多标签统计信息
   - 每个标签的点数
   - 每个标签的平均距离
   - 覆盖率
   - 改进幅度

### result_dict 返回值
```python
{
    'best_params': 最优参数,
    'initial_params': 初始参数,
    'alpha_deg': α角度（度）,
    'beta_deg': β角度（度）,
    'gamma_deg': γ角度（度）,
    'translation': 平移向量 [tx, ty, tz],
    'scaling': 缩放因子 [sx, sy],
    'final_cost': 最终代价（mm）,
    'num_iterations': 迭代次数,
    'elapsed_time': 耗时（秒）,
    'best_slice': 最优切片图像,
    'us_label_points': US各标签边缘点,
    'initial_ct_label_points': 初始CT各标签边缘点,
    'final_ct_label_points': 最终CT各标签边缘点,
    'labels': 使用的标签列表
}
```

## 实用技巧

### 1. 权重设置建议
- **等权重** (默认): 适合四腔都重要的场景
- **强调左心** ({1:2.0, 2:2.0, 3:1.0, 4:1.0}): 适合左心更重要的临床场景
- **仅左心** (labels=[1,2]): 适合只关注左心的应用

### 2. 优化器选择
- **Powell** (use_gps=False): 
  - 更稳定，推荐用于多标签配准
  - 收敛速度较快
  - 适合参数空间较平滑的情况

- **GPS** (use_gps=True):
  - 对噪声更鲁棒
  - 可能需要更多迭代
  - 适合目标函数不光滑的情况

### 3. 参数调优建议
- `max_correspondence_dist`: 根据初始对齐质量调整
  - 初始对齐好: 20-30mm
  - 初始对齐差: 40-50mm
  
- `inner_opt_iterations`: 根据计算时间权衡
  - 快速测试: 10-20
  - 精细配准: 30-50
  
- `min_iterations`: 避免过早停止
  - 一般设置: 5-10
  - 精细配准: 10-20

## 示例代码

完整的示例代码请参考 `multi_label_icp_example.py`，其中包含：
1. 等权重配准示例
2. 自定义权重配准示例
3. 部分标签配准示例
4. 结果对比

## 性能优化

### 加速技巧
1. 减少 `subsample` 参数值可以增加边缘点数量，提高精度但降低速度
2. 使用 `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` 控制线程数
3. 禁用警告显示: `sitk.ProcessObject_SetGlobalWarningDisplay(False)`

### 内存优化
- 大尺寸图像可以适当增加 `subsample` 参数
- 批处理时及时清理中间结果

## 常见问题

### Q1: 某个标签没有边缘点怎么办？
**A**: 代码会自动跳过该标签，并在控制台输出警告信息。确保你的mask标注正确。

### Q2: 收敛太慢怎么办？
**A**: 
- 检查初始参数是否合理
- 适当增加 `max_correspondence_dist`
- 减少 `min_iterations`

### Q3: 配准结果不理想？
**A**:
- 检查初始对齐是否合理
- 尝试调整标签权重
- 增加 `inner_opt_iterations`
- 尝试不同的优化器（Powell vs GPS）

### Q4: 如何判断哪种权重方案最好？
**A**: 
- 查看 `multi_label_stats.txt` 中各标签的距离
- 可视化结果，观察哪些心腔对齐得更好
- 根据临床需求选择重点心腔

## 与原始ICP的对比

| 特性 | 原始ICP | 多标签ICP |
|------|---------|-----------|
| 匹配精度 | 一般 | 高 |
| 错误匹配 | 可能 | 避免 |
| 可解释性 | 低 | 高 |
| 灵活性 | 低 | 高 |
| 计算速度 | 快 | 稍慢 |

## 贡献与反馈

如有问题或建议，欢迎提交Issue或Pull Request。

