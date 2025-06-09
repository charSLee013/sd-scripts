import os
import json
import re
from typing import Dict, List, Tuple
from collections import defaultdict

# 基于分析结果的标签本体论定义
TAG_CATEGORIES = {
    "meta": ["photograph", "image", "shot", "architectural photography"],
    
    "quality": ["professional", "high quality", "detailed", "sharp", "clear", 
               "elegant", "sophisticated", "stylish", "luxurious"],
    
    "style": ["modern interior", "contemporary style", "contemporary design", 
              "minimalist", "minimalist design", "minimalistic", "minimalist decor",
              "contemporary", "modern", "modern architecture", "modern design",
              "industrial style", "mid-century modern", "sleek design"],
    
    "space": ["indoor", "living room", "modern living room", "bedroom", "modern bedroom",
              "kitchen", "modern kitchen", "office", "modern office", "bathroom", 
              "modern bathroom", "interior design", "empty room", "apartment",
              "interior", "home decor"],
    
    "lighting": ["natural light", "soft lighting", "recessed lighting", "sunlight",
                 "warm lighting", "daytime", "ceiling lights", "ambient lighting",
                 "task lighting", "accent lighting"],
    
    "colors": ["neutral colors", "white walls", "beige walls", "white curtains",
               "warm tones", "warm colors", "cool tones", "monochromatic",
               "earth tones", "muted colors"],
    
    "materials": ["wooden floor", "wooden table", "wooden shelves", "glass doors",
                  "wooden coffee table", "marble floor", "glass coffee table",
                  "glass walls", "concrete floor", "wooden ceiling", "wooden chair",
                  "marble countertop", "textured fabric"],
    
    "furniture": ["furniture", "sleek furniture", "contemporary furniture", 
                  "modern furniture", "comfortable", "round table", "bookshelves",
                  "shelves", "armchair", "desk", "sofa", "chair"],
    
    "atmosphere": ["cozy atmosphere", "cozy", "clean lines", "sleek", "clean",
                   "spacious", "serene atmosphere", "calm atmosphere", "peaceful",
                   "relaxing", "inviting"],
    
    "architectural": ["large windows", "large window", "high ceiling", "glass door",
                      "glass window", "open floor plan", "vaulted ceiling",
                      "exposed beams", "floor-to-ceiling windows"],
    
    "decorative": ["books", "potted plant", "abstract art", "indoor plants",
                   "artwork", "decorative objects", "vase", "flowers", "plants",
                   "greenery"],
    
    "composition": ["shadows", "geometric shapes", "symmetry", "asymmetrical",
                    "vertical composition", "horizontal composition", "perspective",
                    "depth of field", "framing"],
    
    "environment": ["urban setting", "cityscape view", "urban view", "city view",
                    "outdoor view", "ocean view", "nature view", "garden view",
                    "balcony", "terrace"]
}

def load_analysis_results(json_file: str) -> Dict:
    """加载标签分析结果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_tag_mapping(analysis_results: Dict) -> Dict[str, str]:
    """创建标签到类别的映射"""
    tag_to_category = {}
    
    # 从分析结果中提取所有分类的标签
    for category, tags in analysis_results['categories'].items():
        for tag in tags.keys():
            tag_to_category[tag] = category
    
    # 添加手动定义的映射
    for category, tags in TAG_CATEGORIES.items():
        for tag in tags:
            tag_to_category[tag] = category
    
    return tag_to_category

def categorize_tags_advanced(tags: List[str], tag_mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """高级标签分类，支持模糊匹配和语义理解"""
    categorized = defaultdict(list)
    
    for tag in tags:
        tag_lower = tag.lower().strip()
        category_found = False
        
        # 精确匹配
        if tag_lower in tag_mapping:
            categorized[tag_mapping[tag_lower]].append(tag)
            category_found = True
        else:
            # 模糊匹配
            for category, category_tags in TAG_CATEGORIES.items():
                for category_tag in category_tags:
                    if category_tag.lower() in tag_lower or tag_lower in category_tag.lower():
                        categorized[category].append(tag)
                        category_found = True
                        break
                if category_found:
                    break
        
        # 如果没有找到匹配，进行语义推理
        if not category_found:
            category = semantic_categorize(tag_lower)
            categorized[category].append(tag)
    
    return dict(categorized)

def semantic_categorize(tag: str) -> str:
    """基于语义规则的标签分类"""
    tag_lower = tag.lower()
    
    # 颜色相关
    color_keywords = ['white', 'black', 'gray', 'grey', 'brown', 'beige', 'blue', 'green', 'red', 'yellow']
    if any(color in tag_lower for color in color_keywords):
        return 'colors'
    
    # 材质相关
    material_keywords = ['wood', 'glass', 'metal', 'stone', 'concrete', 'marble', 'leather', 'fabric']
    if any(material in tag_lower for material in material_keywords):
        return 'materials'
    
    # 空间相关
    space_keywords = ['room', 'kitchen', 'bedroom', 'bathroom', 'office', 'living', 'dining']
    if any(space in tag_lower for space in space_keywords):
        return 'space'
    
    # 光照相关
    lighting_keywords = ['light', 'lighting', 'bright', 'dark', 'shadow', 'illumination']
    if any(light in tag_lower for light in lighting_keywords):
        return 'lighting'
    
    # 家具相关
    furniture_keywords = ['chair', 'table', 'sofa', 'bed', 'desk', 'shelf', 'cabinet', 'furniture']
    if any(furniture in tag_lower for furniture in furniture_keywords):
        return 'furniture'
    
    # 装饰相关
    decor_keywords = ['art', 'plant', 'flower', 'book', 'vase', 'decoration', 'painting']
    if any(decor in tag_lower for decor in decor_keywords):
        return 'decorative'
    
    # 默认分类
    return 'general'

def create_structured_prompt(categorized_tags: Dict[str, List[str]]) -> str:
    """创建NovelAI V3风格的结构化提示词"""
    
    # 定义类别优先级和显示名称
    category_order = [
        ('meta', '元数据'),
        ('quality', '质量'),
        ('style', '风格'),
        ('space', '空间'),
        ('lighting', '光照'),
        ('colors', '色彩'),
        ('materials', '材质'),
        ('furniture', '家具'),
        ('atmosphere', '氛围'),
        ('architectural', '建筑'),
        ('decorative', '装饰'),
        ('composition', '构图'),
        ('environment', '环境'),
        ('general', '通用')
    ]
    
    structured_parts = []
    
    for category_key, category_name in category_order:
        if category_key in categorized_tags and categorized_tags[category_key]:
            # 去重并限制数量
            unique_tags = list(dict.fromkeys(categorized_tags[category_key]))[:8]
            tags_str = ', '.join(unique_tags)
            structured_parts.append(f"<|{category_key}|>: {tags_str}")
    
    return ', '.join(structured_parts)

def process_single_file(file_path: str, tag_mapping: Dict[str, str]) -> Tuple[str, str]:
    """处理单个标签文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 1:
            return "", ""
        
        # 解析原始标签
        original_tags_line = lines[0].strip()
        original_tags = [tag.strip() for tag in original_tags_line.split(',') if tag.strip()]
        
        # 保留自然语言描述
        description = lines[1].strip() if len(lines) > 1 else ""
        
        # 分类标签
        categorized_tags = categorize_tags_advanced(original_tags, tag_mapping)
        
        # 生成结构化提示词
        structured_prompt = create_structured_prompt(categorized_tags)
        
        return structured_prompt, description
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return "", ""

def process_dataset(data_dir: str, output_dir: str, tag_mapping: Dict[str, str]):
    """处理整个数据集"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stats = {
        'processed': 0,
        'failed': 0,
        'avg_original_tags': 0,
        'avg_structured_length': 0
    }
    
    total_original_tags = 0
    total_structured_length = 0
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(data_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            structured_prompt, description = process_single_file(input_path, tag_mapping)
            
            if structured_prompt:
                # 保存重构后的文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(structured_prompt + '\n')
                    if description:
                        f.write(description + '\n')
                
                stats['processed'] += 1
                
                # 统计信息
                original_tag_count = len(open(input_path, 'r').readline().split(','))
                total_original_tags += original_tag_count
                total_structured_length += len(structured_prompt)
                
                # 显示进度
                if stats['processed'] % 100 == 0:
                    print(f"已处理 {stats['processed']} 个文件...")
                    
            else:
                stats['failed'] += 1
    
    # 计算统计信息
    if stats['processed'] > 0:
        stats['avg_original_tags'] = total_original_tags / stats['processed']
        stats['avg_structured_length'] = total_structured_length / stats['processed']
    
    return stats

def verify_token_length(prompt: str, max_tokens: int = 225) -> bool:
    """验证提示词token长度（简化版本）"""
    # 简化的token计算：大概每4.5个字符=1个token
    estimated_tokens = len(prompt) / 4.5
    return estimated_tokens <= max_tokens

def main():
    """主函数"""
    data_dir = "/root/data/cluster_4"
    output_dir = "/root/data/cluster_4_restructured"
    analysis_file = "tag_analysis_results.json"
    
    print("=== NovelAI V3风格数据重构开始 ===")
    
    # 加载分析结果
    print("1. 加载标签分析结果...")
    analysis_results = load_analysis_results(analysis_file)
    
    # 创建标签映射
    print("2. 创建标签类别映射...")
    tag_mapping = create_tag_mapping(analysis_results)
    print(f"   映射了 {len(tag_mapping)} 个标签")
    
    # 处理数据集
    print("3. 开始处理数据集...")
    stats = process_dataset(data_dir, output_dir, tag_mapping)
    
    # 显示结果
    print("\n=== 处理完成 ===")
    print(f"成功处理: {stats['processed']} 个文件")
    print(f"处理失败: {stats['failed']} 个文件")
    print(f"平均原始标签数: {stats['avg_original_tags']:.1f}")
    print(f"平均结构化长度: {stats['avg_structured_length']:.1f} 字符")
    print(f"输出目录: {output_dir}")
    
    # 测试几个样本的token长度
    print("\n=== Token长度验证 ===")
    sample_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')][:5]
    for filename in sample_files:
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt = f.readline().strip()
        
        token_ok = verify_token_length(prompt)
        print(f"{filename}: {len(prompt)} 字符, Token估计: {len(prompt)/4.5:.0f}, 状态: {'✓' if token_ok else '✗'}")

if __name__ == "__main__":
    main() 