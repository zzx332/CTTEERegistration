"""
多标签ICP配准使用示例
演示如何使用 register_icp_multi_label 方法进行分标签配准
"""

import numpy as np
from pathlib import Path
# import 2D_3D_registration_boundary_gps 
from .2D_3D_registration_boundary_gps import TwoD_ThreeD_Registration

def main():
    """主函数"""
    
    # 初始化配准器
    registrator = TwoD_ThreeD_Registration()
   
    # 加载数据（包括mask）
    ct_path = r"D:\dataset\CT\MM-WHS2017\ct_train\ct_train_1004_image.nii"
    ultrasound_path = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_image_initial_transform.nii.gz"
    ct_mask_path = r"D:\dataset\CT\MM-WHS2017\ct_train\ct_train_1004_remapped_label.nii"  # CT心脏mask路径
    us_mask_path = r"D:\dataset\TEECT_data\tee\patient-1-4\slice_060_label_initial_transform.nii.gz"
    output_dir = r"D:\dataset\TEECT_data\registration_results\2d_3d_multi_label"
    
    registrator.load_images(ct_path, ultrasound_path, ct_mask_path, us_mask_path)
    
    # 初始参数
    initial_params = np.array([
        np.radians(50),   # alpha: 50°
        np.radians(-25),  # beta: -25°
        np.radians(0),    # gamma: 0°
        5.77, -163.36, 5.0,   # translation: (x, y, z) mm
        1.0, 1.0         # scaling: (sx, sy)
    ])
    
    # ===== 方法1: 使用默认权重（所有标签等权） =====
    print("\n" + "="*80)
    print("方法1: 默认权重（所有标签等权）")
    print("="*80)
    
    best_params_1, result_dict_1 = registrator.register_icp_multi_label(
        initial_params=initial_params,
        max_iterations=20,  # ICP外层迭代
        tolerance=0.1,  # 收敛阈值：平均距离变化<0.1mm
        max_correspondence_dist=30.0,  # 最大对应距离30mm
        inner_opt_iterations=30,  # 每次ICP迭代内部优化30次
        min_iterations=5,  # 最小迭代次数（避免过早停止）
        use_gps=False,  # 使用Powell而非GPS（更稳定）
        labels=[1, 2, 3, 4],  # 使用所有四腔标签
        label_weights=None  # 默认权重
    )
    
    # 可视化结果
    registrator.visualize_result(result_dict_1, output_dir + "_equal_weight")
    registrator.visualize_multi_label_correspondences(
        result_dict_1, 
        output_dir + "_equal_weight",
        max_correspondence_dist=30.0
    )
    
    # # ===== 方法2: 自定义权重（强调左心室和左心房） =====
    # print("\n" + "="*80)
    # print("方法2: 自定义权重（强调左心室和左心房）")
    # print("="*80)
    
    # # 重新加载图像（重置状态）
    # registrator.load_images(ct_path, ultrasound_path, ct_mask_path, us_mask_path)
    
    # best_params_2, result_dict_2 = registrator.register_icp_multi_label(
    #     initial_params=initial_params,
    #     max_iterations=20,
    #     tolerance=0.1,
    #     max_correspondence_dist=30.0,
    #     inner_opt_iterations=30,
    #     min_iterations=5,
    #     use_gps=False,
    #     labels=[1, 2, 3, 4],
    #     label_weights={
    #         1: 2.0,  # LV - 左心室（权重加倍）
    #         2: 2.0,  # LA - 左心房（权重加倍）
    #         3: 1.0,  # RV - 右心室（正常权重）
    #         4: 1.0   # RA - 右心房（正常权重）
    #     }
    # )
    
    # # 可视化结果
    # registrator.visualize_result(result_dict_2, output_dir + "_weighted")
    # registrator.visualize_multi_label_correspondences(
    #     result_dict_2, 
    #     output_dir + "_weighted",
    #     max_correspondence_dist=30.0
    # )
    
    # # ===== 方法3: 仅使用部分标签（只配准左心室和左心房） =====
    # print("\n" + "="*80)
    # print("方法3: 仅使用部分标签（只配准左心室和左心房）")
    # print("="*80)
    
    # # 重新加载图像
    # registrator.load_images(ct_path, ultrasound_path, ct_mask_path, us_mask_path)
    
    # best_params_3, result_dict_3 = registrator.register_icp_multi_label(
    #     initial_params=initial_params,
    #     max_iterations=20,
    #     tolerance=0.1,
    #     max_correspondence_dist=30.0,
    #     inner_opt_iterations=30,
    #     min_iterations=5,
    #     use_gps=False,
    #     labels=[1, 2],  # 仅使用LV和LA
    #     label_weights=None
    # )
    
    # # 可视化结果
    # registrator.visualize_result(result_dict_3, output_dir + "_lv_la_only")
    # registrator.visualize_multi_label_correspondences(
    #     result_dict_3, 
    #     output_dir + "_lv_la_only",
    #     max_correspondence_dist=30.0
    # )
    
    # # ===== 对比结果 =====
    # print("\n" + "="*80)
    # print("配准结果对比")
    # print("="*80)
    
    # print(f"\n方法1（等权）:")
    # print(f"  最终代价: {result_dict_1['final_cost']:.3f}mm")
    # print(f"  迭代次数: {result_dict_1['num_iterations']}")
    # print(f"  耗时: {result_dict_1['elapsed_time']:.1f}秒")
    
    # print(f"\n方法2（加权）:")
    # print(f"  最终代价: {result_dict_2['final_cost']:.3f}mm")
    # print(f"  迭代次数: {result_dict_2['num_iterations']}")
    # print(f"  耗时: {result_dict_2['elapsed_time']:.1f}秒")
    
    # print(f"\n方法3（LV+LA）:")
    # print(f"  最终代价: {result_dict_3['final_cost']:.3f}mm")
    # print(f"  迭代次数: {result_dict_3['num_iterations']}")
    # print(f"  耗时: {result_dict_3['elapsed_time']:.1f}秒")
    
    # print("\n配准完成！所有结果已保存。")


if __name__ == "__main__":
    main()

