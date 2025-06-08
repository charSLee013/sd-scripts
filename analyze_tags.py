import os
import re
from collections import Counter
import json

def analyze_dataset_tags(data_dir):
    """分析数据集中的标签分布和词频"""
    all_tags = []
    tag_stats = {
        'total_files': 0,
        'valid_files': 0,
        'tag_counts': Counter(),
        'avg_tags_per_file': 0,
        'sample_tags': []
    }
    
    # 遍历所有txt文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            tag_stats['total_files'] += 1
            file_path = os.path.join(data_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                if len(lines) >= 1:
                    # 第一行是标签
                    tags_line = lines[0].strip()
                    
                    # 分割标签（以逗号分隔）
                    tags = [tag.strip() for tag in tags_line.split(',') if tag.strip()]
                    
                    if tags:
                        tag_stats['valid_files'] += 1
                        all_tags.extend(tags)
                        tag_stats['tag_counts'].update(tags)
                        
                        # 保存前10个文件的标签作为样本
                        if len(tag_stats['sample_tags']) < 10:
                            tag_stats['sample_tags'].append({
                                'file': filename,
                                'tags': tags,
                                'count': len(tags)
                            })
                            
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
    
    # 计算统计信息
    if tag_stats['valid_files'] > 0:
        tag_stats['avg_tags_per_file'] = len(all_tags) / tag_stats['valid_files']
    
    return tag_stats

def categorize_tags(tag_counts, min_frequency=5):
    """基于词频和语义特征对标签进行初步分类"""
    categories = {
        'photography_technical': [],    # 摄影技术词汇
        'architecture_style': [],      # 建筑风格
        'materials_textures': [],      # 材质纹理
        'colors': [],                  # 颜色相关
        'spatial_elements': [],        # 空间元素
        'lighting': [],                # 光照相关
        'furniture_objects': [],       # 家具物品
        'atmosphere_mood': [],         # 氛围情绪
        'technical_quality': [],       # 技术质量
        'compositional': [],           # 构图相关
        'uncategorized': []            # 未分类
    }
    
    # 定义分类关键词
    category_keywords = {
        'photography_technical': ['photograph', 'image', 'shot', 'camera', 'lens', 'depth of field', 'bokeh'],
        'architecture_style': ['modern', 'contemporary', 'minimalist', 'industrial', 'architecture', 'building', 'structure'],
        'materials_textures': ['wooden', 'glass', 'metal', 'concrete', 'stone', 'fabric', 'leather', 'marble'],
        'colors': ['white', 'black', 'gray', 'grey', 'brown', 'beige', 'neutral colors', 'dark', 'light'],
        'spatial_elements': ['room', 'space', 'interior', 'indoor', 'outdoor', 'wall', 'ceiling', 'floor'],
        'lighting': ['natural light', 'lighting', 'lights', 'recessed lighting', 'ceiling lights', 'daylight', 'sunlight'],
        'furniture_objects': ['chair', 'table', 'sofa', 'furniture', 'shelves', 'cabinet', 'bed', 'desk'],
        'atmosphere_mood': ['clean', 'sleek', 'elegant', 'cozy', 'spacious', 'bright', 'warm', 'empty', 'unoccupied'],
        'technical_quality': ['high quality', 'detailed', 'sharp', 'clear', 'professional', 'high resolution'],
        'compositional': ['symmetry', 'balance', 'perspective', 'angle', 'view', 'composition', 'layout']
    }
    
    # 对每个标签进行分类
    for tag, count in tag_counts.most_common():
        if count >= min_frequency:
            categorized = False
            tag_lower = tag.lower()
            
            for category, keywords in category_keywords.items():
                if any(keyword in tag_lower for keyword in keywords):
                    categories[category].append((tag, count))
                    categorized = True
                    break
            
            if not categorized:
                categories['uncategorized'].append((tag, count))
    
    return categories

def generate_report(tag_stats, categories):
    """生成分析报告"""
    print("=== 数据集标签分析报告 ===")
    print(f"总文件数: {tag_stats['total_files']}")
    print(f"有效文件数: {tag_stats['valid_files']}")
    print(f"平均每文件标签数: {tag_stats['avg_tags_per_file']:.2f}")
    print(f"唯一标签总数: {len(tag_stats['tag_counts'])}")
    print()
    
    print("=== 高频标签 TOP 20 ===")
    for tag, count in tag_stats['tag_counts'].most_common(20):
        print(f"{tag}: {count}")
    print()
    
    print("=== 标签分类统计 ===")
    for category, tags in categories.items():
        if tags:
            print(f"\n{category.upper()} ({len(tags)} 个标签):")
            for tag, count in tags[:10]:  # 只显示前10个
                print(f"  {tag}: {count}")
    
    print("\n=== 样本文件标签 ===")
    for sample in tag_stats['sample_tags'][:5]:
        print(f"\n文件: {sample['file']}")
        print(f"标签数: {sample['count']}")
        print(f"标签: {', '.join(sample['tags'][:10])}{'...' if len(sample['tags']) > 10 else ''}")

if __name__ == "__main__":
    data_dir = "/root/data/cluster_4"
    
    print("开始分析数据集标签...")
    tag_stats = analyze_dataset_tags(data_dir)
    
    print("对标签进行分类...")
    categories = categorize_tags(tag_stats['tag_counts'])
    
    print("生成分析报告...")
    generate_report(tag_stats, categories)
    
    # 保存详细结果到JSON文件
    results = {
        'statistics': {
            'total_files': tag_stats['total_files'],
            'valid_files': tag_stats['valid_files'],
            'avg_tags_per_file': tag_stats['avg_tags_per_file'],
            'unique_tags': len(tag_stats['tag_counts'])
        },
        'top_tags': dict(tag_stats['tag_counts'].most_common(50)),
        'categories': {k: dict(v) for k, v in categories.items()},
        'samples': tag_stats['sample_tags']
    }
    
    with open('tag_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n详细结果已保存到 tag_analysis_results.json") 