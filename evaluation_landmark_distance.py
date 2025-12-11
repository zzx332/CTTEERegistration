import numpy as np  
import SimpleITK as sitk
from typing import Optional, Tuple
from LV_apex_cal import calculate_landmark

def calculate_projection_length(point_a, point_b, normal_vector):
    """
    计算线段AB在法向量n上的投影长度。

    Args:
        point_a (np.ndarray): 线段的起始点坐标 (例如: [x, y, z])。
        point_b (np.ndarray): 线段的结束点坐标 (例如: [x, y, z])。
        normal_vector (np.ndarray): 法向量 (例如: [nx, ny, nz])。

    Returns:
        float: 投影长度。
    """
    # 确保输入是numpy数组
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    normal_vector = np.array(normal_vector)

    # 1. 计算线段向量 AB
    vector_ab = point_b - point_a

    # 2. 计算法向量的模长
    magnitude_n = np.linalg.norm(normal_vector)

    # 检查法向量模长是否为零，避免除以零错误
    if magnitude_n == 0:
        raise ValueError("法向量不能为零向量。")

    # 3. 计算点积 (vector_ab . normal_vector)
    dot_product = np.dot(vector_ab, normal_vector)

    # 4. 计算投影长度
    projection_length = dot_product / magnitude_n

    return projection_length


def orthogonalize_direction_2d(self, direction):
    """
    direction: 长度为 4 的 tuple/list，对应 2×2 矩阵 (row-major)
    返回新的正交方向（仍然是 row-major）
    """
    d = np.array(direction).reshape(2,2)

    # 第 1 列
    e1 = d[:,0]
    e1 = e1 / np.linalg.norm(e1)

    # 第 2 列，正交化： e2 = e2 - (e2·e1)*e1
    e2 = d[:,1]
    e2 = e2 - np.dot(e2, e1) * e1
    e2 = e2 / np.linalg.norm(e2)

    # 重组为 2×2 row-major
    d_new = np.column_stack((e1, e2)).reshape(-1)
    return tuple(d_new)
    
def cal_apex_distance(
    ct_mask: sitk.Image,
    alpha: float,
    beta: float,
    gamma: float,
    tx: float,
    ty: float,
    tz: float,
    sx: float = 1.0,
    sy: float = 1.0
):
    slice_size = (512, 512)
    slice_spacing = (0.5, 0.5)
    # ===== 修复1：确保spacing始终为正 =====
    slice_spacing_scaled = [
        abs(slice_spacing[0] / sx),  # 取绝对值
        abs(slice_spacing[1] / sy),  # 取绝对值
        1.0
    ]
    
    # 构建旋转矩阵（3x3）
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_g, sin_g = np.cos(gamma), np.sin(gamma)
    
    Rx = np.array([
        [1,      0,       0     ],
        [0,      cos_a,  -sin_a],
        [0,      sin_a,   cos_a]
    ])

    Ry = np.array([
        [cos_b,  0,       sin_b],
        [0,      1,       0    ],
        [-sin_b, 0,       cos_b]
    ])

    Rz = np.array([
        [cos_g, -sin_g,  0],
        [sin_g,  cos_g,  0],
        [0,      0,      1]
    ])

    # 组合旋转
    R = Rz @ Ry @ Rx
    
    # ===== 修复2：验证旋转矩阵的正交性和行列式 =====
    det_R = np.linalg.det(R)
    orthogonality_error = np.max(np.abs(R @ R.T - np.eye(3)))
    
   
    # 如果行列式为负，说明有镜像，需要修正
    if det_R < 0:
        print(f"  ⚠️ 警告：旋转矩阵行列式为负 ({det_R:.6f})，进行修正...")
        # 翻转Z轴方向
        R[:, 2] = -R[:, 2]
        det_R = np.linalg.det(R)
        print(f"  修正后行列式: {det_R:.6f}")
    
    # 如果正交性误差太大，重新正交化
    if orthogonality_error > 1e-6:
        print(f"  ⚠️ 警告：旋转矩阵不正交，进行Gram-Schmidt正交化...")
        # Gram-Schmidt正交化
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]
        
        # 正交化
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)  # 确保右手系
        
        R = np.column_stack([x_axis, y_axis, z_axis])
        print(f"  正交化后行列式: {np.linalg.det(R):.6f}")

    # CT中心点
    ct_size = np.array(ct_mask.GetSize())
    ct_spacing = np.array(ct_mask.GetSpacing())
    ct_origin = np.array(ct_mask.GetOrigin())
    # 适应不同方向
    ct_direction = np.array(ct_mask.GetDirection()).reshape(3, 3)    
    ct_center = ct_origin + ct_direction @ ((ct_size - 1) * ct_spacing / 2.0)
    # ct_center = ct_origin +  ((ct_size - 1) * ct_spacing / 2.0)
    # R_ct_space = ct_direction @ R @ ct_direction.T
    # 切片中心（在3D空间中，经过旋转和平移后）
    slice_center = ct_center + np.array([tx, ty, tz])
    # slice_center_volume = ct_mask.TransformPhysicalPointToIndex(slice_center)
    # 切片的方向向量（旋转后的坐标轴）
    x_axis = R[:, 0]  # 切片的X轴方向（列方向）
    y_axis = R[:, 1]  # 切片的Y轴方向（行方向）
    z_axis = R[:, 2]  # 切片的法向量（Z轴方向）
    label_dict = {
        "rv_label": 1,
        "ra_label": 2,
        "lv_label": 4,
        "la_label": 3
    }
    landmark = calculate_landmark(sitk.GetArrayFromImage(ct_mask), label_dict)
    apex_physical = ct_mask.TransformContinuousIndexToPhysicalPoint(landmark["apex"])
    mitral_center_physical = ct_mask.TransformContinuousIndexToPhysicalPoint(landmark["mitral_center"])
    tricuspid_annulus_center_physical = ct_mask.TransformContinuousIndexToPhysicalPoint(landmark["tricuspid_annulus_center"])
    apex_distance = calculate_projection_length(apex_physical, slice_center, z_axis)
    mitral_distance = calculate_projection_length(mitral_center_physical, slice_center, z_axis)
    tricuspid_annulus_distance = calculate_projection_length(tricuspid_annulus_center_physical, slice_center, z_axis)
    
    # # 方法1: 先提取变量（推荐）
    # apex_coord = landmark["apex"]
    # mitral_coord = landmark["mitral_center"]
    # tricuspid_coord = landmark["tricuspid_annulus_center"]
    
    # print(f"apex: {apex_coord}, mitral_center: {mitral_coord}, tricuspid_annulus_center: {tricuspid_coord}, \
    #       \napex_distance: {apex_distance:.2f}, mitral_distance: {mitral_distance:.2f}, tricuspid_annulus_distance: {tricuspid_annulus_distance:.2f}")
    return abs(apex_distance), abs(mitral_distance), abs(tricuspid_annulus_distance)

if __name__ == "__main__":
    alpha = np.radians(52.64)
    beta = np.radians(-47.34)
    gamma = np.radians(-2.67)
    tx = 1
    ty = 8
    tz = 11.5
    sx = 0.8
    sy = 1.2
    ct_mask_path = r"D:\dataset\Cardiac_Multi-View_US-CT_Paired_Dataset\Segmentation\Patient_0036\Patient_0036_label.nii.gz"
    ct_mask = sitk.ReadImage(ct_mask_path)
    cal_apex_distance(ct_mask, alpha, beta, gamma, tx, ty, tz, sx, sy)