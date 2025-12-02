#!/usr/bin/env bash
set -euo pipefail

# 按 WSL 的方式映射 Windows 磁盘路径，D: -> /mnt/d
INPUT_DIR="/mnt/d/dataset/Cardiac_Multi-View_US-CT_Paired_Dataset/CT_resampled_nii"
OUTPUT_DIR="/mnt/d/dataset/Cardiac_Multi-View_US-CT_Paired_Dataset/Segmentation"

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.nii.gz; do
    # 防止目录下没有匹配文件时报错
    [ -e "$f" ] || continue

    fname=$(basename "$f" .nii.gz)
    echo "fullpath $f ..."
    echo "Processing $fname ..."
    TotalSegmentator -i "$f" -o "$OUTPUT_DIR/$fname" -ta heartchambers_highres
done
