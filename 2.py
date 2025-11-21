import numpy as np

alpha = np.deg2rad(50.0)  # 绕X轴
beta = np.deg2rad(-25.0)  # 绕Y轴

# 标准旋转矩阵
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(alpha), -np.sin(alpha)],
    [0, np.sin(alpha), np.cos(alpha)]
])

Ry = np.array([
    [np.cos(beta), 0, np.sin(beta)],
    [0, 1, 0],
    [-np.sin(beta), 0, np.cos(beta)]
])

print("Rx(50°) =")
print(Rx)
print("\nRy(-25°) =")
print(Ry)

# 两种组合顺序
R_yx = Ry @ Rx  # 先X后Y
R_xy = Rx @ Ry  # 先Y后X

print("\n" + "="*60)
print("Ry @ Rx (先绕X轴50°，再绕Y轴-25°) =")
print(R_yx)

print("\nRx @ Ry (先绕Y轴-25°，再绕X轴50°) =")
print(R_xy)

print("\n" + "="*60)
print("对比你的结果：")
print("\nR1 (方法1) =")
R1 = np.array([
    [ 0.90630779, -0.        , -0.42261826],
    [-0.32374437,  0.64278761, -0.69427204],
    [ 0.27165378,  0.76604444,  0.58256342]
])
print(R1)

print("\nR2 (方法2) =")
R2 = np.array([
    [ 0.90630779, -0.32374437, -0.27165378],
    [ 0.        ,  0.64278761, -0.76604444],
    [ 0.42261826,  0.69427204,  0.58256342]
])
print(R2)

print("\n匹配检查：")
print(f"R1 == Ry @ Rx ? {np.allclose(R1, R_yx)}")
print(f"R1 == Rx @ Ry ? {np.allclose(R1, R_xy)}")
print(f"R2 == Ry @ Rx ? {np.allclose(R2, R_yx)}")
print(f"R2 == Rx @ Ry ? {np.allclose(R2, R_xy)}")