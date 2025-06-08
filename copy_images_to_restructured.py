#!/usr/bin/env python3
"""
将原始数据目录中的图片文件复制到重构后的目录中
确保图片和重构后的标签文件配对
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def copy_images_to_restructured():
    # 原始数据目录
    source_dir = Path("/root/data/cluster_4")
    # 重构后的数据目录
    target_dir = Path("/root/data/cluster_4_restructured_v2")
    
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    
    # 确保目标目录存在
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有重构后的标签文件
    txt_files = list(target_dir.glob("*.txt"))
    print(f"找到 {len(txt_files)} 个重构后的标签文件")
    
    # 图片扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
    
    copied_count = 0
    missing_count = 0
    
    for txt_file in tqdm(txt_files, desc="复制图片文件"):
        # 获取基础文件名（不含扩展名）
        base_name = txt_file.stem
        
        # 在源目录中寻找对应的图片文件
        image_found = False
        for ext in image_extensions:
            source_image = source_dir / f"{base_name}{ext}"
            if source_image.exists():
                target_image = target_dir / f"{base_name}{ext}"
                
                # 如果目标图片不存在，则复制
                if not target_image.exists():
                    shutil.copy2(source_image, target_image)
                    copied_count += 1
                
                image_found = True
                break
        
        if not image_found:
            print(f"警告: 未找到 {base_name} 对应的图片文件")
            missing_count += 1
    
    print(f"\n复制完成!")
    print(f"成功复制: {copied_count} 个图片文件")
    print(f"缺失图片: {missing_count} 个文件")
    
    # 验证最终结果
    final_txt_count = len(list(target_dir.glob("*.txt")))
    final_img_count = 0
    for ext in image_extensions:
        final_img_count += len(list(target_dir.glob(f"*{ext}")))
    
    print(f"\n最终统计:")
    print(f"标签文件: {final_txt_count}")
    print(f"图片文件: {final_img_count}")
    print(f"配对率: {final_img_count/final_txt_count*100:.1f}%")

if __name__ == "__main__":
    copy_images_to_restructured() 