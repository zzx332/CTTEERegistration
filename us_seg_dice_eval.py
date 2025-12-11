import os
import SimpleITK as sitk
import numpy as np

pred_path = r"D:\dataset\nnunet_data\Dataset001\predictions"
label_path = r"D:\dataset\nnunet_data\Dataset001\labelsTs"
dice_scores = {3: 0, 4: 0}
for file in os.listdir(label_path):
    pred_file = os.path.join(pred_path, file)
    label_file = os.path.join(label_path, file)
    pred_img = sitk.ReadImage(pred_file)
    label_img = sitk.ReadImage(label_file)
    pred_arr = sitk.GetArrayFromImage(pred_img)
    label_arr = sitk.GetArrayFromImage(label_img)
    for label_id in [3,4]:
        pred_mask = (pred_arr == label_id)
        label_mask = (label_arr == label_id)
        intersection = np.sum(pred_mask & label_mask)
        union = np.sum(pred_mask) + np.sum(label_mask)
        if union > 0:
            dice = 2.0 * intersection / union
        else:
            dice = 0.0
        dice_scores[label_id] += dice
    print(file)
print(f"Dice scores: {dice_scores[3]/len(os.listdir(label_path)):.4f}, {dice_scores[4]/len(os.listdir(label_path)):.4f}")
# print(f"Dice scores for label 3: {dice_scores[3]/len(os.listdir(label_path)):.4f}, Dice scores for label 4: {dice_scores[4]/len(os.listdir(label_path)):.4f}")
# print(f"Mean Dice: {np.mean(list(dice_scores.values())):.4f}")
# print(f"Std Dice: {np.std(list(dice_scores.values())):.4f}")
# print(f"Max Dice: {np.max(list(dice_scores.values())):.4f}")
# print(f"Min Dice: {np.min(list(dice_scores.values())):.4f}")