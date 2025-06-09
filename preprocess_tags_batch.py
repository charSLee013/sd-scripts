import os
import json
import torch
import numpy as np
import argparse
import math
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import CLIPTokenizer, CLIPTextModel
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageOps
import warnings
import sys

# --- 全局配置 ---
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TE1_PATH = "/root/text_encoder/clip-vit-large-patch14"
TE2_PATH = "/root/text_encoder/CLIP-ViT-B-32-laion2B-s34B-b79K"

# 支持的图片格式
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.WEBP', '.BMP', '.TIFF']

# --- 核心标签本体论 (Ontology) ---
TAG_CATEGORIES_ORDER = [
    "meta", "quality", "style", "space", "lighting", "colors", 
    "materials", "furniture", "atmosphere", "architectural", 
    "decorative", "composition", "environment", "general"
]

TAG_ONTOLOGY = {
    "meta": ["photograph", "image", "shot", "architectural photography"],
    "quality": ["professional", "high quality", "detailed", "sharp", "clear", "elegant", "sophisticated", "stylish", "luxurious"],
    "style": ["modern interior", "contemporary style", "contemporary design", "minimalist", "minimalist design", "minimalistic", "minimalist decor", "contemporary", "modern", "modern architecture", "modern design", "industrial style", "mid-century modern", "sleek design"],
    "space": ["indoor", "living room", "bedroom", "kitchen", "office", "bathroom", "interior design", "empty room", "apartment", "interior", "home decor"],
    "lighting": ["natural light", "soft lighting", "recessed lighting", "sunlight", "warm lighting", "daytime", "ceiling lights", "ambient lighting", "task lighting", "accent lighting"],
    "colors": ["neutral colors", "white walls", "beige walls", "white curtains", "warm tones", "warm colors", "cool tones", "monochromatic", "earth tones", "muted colors"],
    "materials": ["wooden floor", "wooden table", "wooden shelves", "glass doors", "wooden coffee table", "marble floor", "glass coffee table", "glass walls", "concrete floor", "wooden ceiling", "wooden chair", "marble countertop", "textured fabric"],
    "furniture": ["furniture", "sleek furniture", "contemporary furniture", "modern furniture", "comfortable", "round table", "bookshelves", "shelves", "armchair", "desk", "sofa", "chair"],
    "atmosphere": ["cozy atmosphere", "cozy", "clean lines", "sleek", "clean", "spacious", "serene atmosphere", "calm atmosphere", "peaceful", "relaxing", "inviting"],
    "architectural": ["large windows", "large window", "high ceiling", "glass door", "glass window", "open floor plan", "vaulted ceiling", "exposed beams", "floor-to-ceiling windows"],
    "decorative": ["books", "potted plant", "abstract art", "indoor plants", "artwork", "decorative objects", "vase", "flowers", "plants", "greenery"],
    "composition": ["shadows", "geometric shapes", "symmetry", "asymmetrical", "vertical composition", "horizontal composition", "perspective", "depth of field", "framing"],
    "environment": ["urban setting", "cityscape view", "urban view", "city view", "outdoor view", "ocean view", "nature view", "garden view", "balcony", "terrace"],
    "general": []
}

class BatchProcessor:
    """批次处理器 - 处理单个批次的文件"""
    
    def __init__(self, max_resolution=5000, output_format="PNG"):
        print(f"[BATCH] 初始化批次处理器 (设备: {DEVICE})")
        self.max_resolution = max_resolution
        self.output_format = output_format
        self._load_models()
        self._build_tag_mapping()

    def _load_models(self):
        """加载TE1和TE2的模型与分词器"""
        try:
            self.te1_tokenizer = CLIPTokenizer.from_pretrained(TE1_PATH)
            self.te1_model = CLIPTextModel.from_pretrained(TE1_PATH).to(DEVICE).eval()
            self.te2_tokenizer = CLIPTokenizer.from_pretrained(TE2_PATH)
            self.te2_model = CLIPTextModel.from_pretrained(TE2_PATH).to(DEVICE).eval()
            print("[BATCH] CLIP模型加载完成")
        except Exception as e:
            print(f"[BATCH ERROR] 模型加载失败: {e}")
            sys.exit(1)

    def _build_tag_mapping(self):
        """构建标签到类别的映射"""
        self.tag_to_category = {}
        for category, tags in TAG_ONTOLOGY.items():
            for tag in tags:
                self.tag_to_category[tag] = category

    @torch.no_grad()
    def _get_embedding(self, text, model_type="te1"):
        """获取文本的嵌入向量"""
        if model_type == "te1":
            tokenizer = self.te1_tokenizer
            model = self.te1_model
        else:
            tokenizer = self.te2_tokenizer
            model = self.te2_model
        
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=77).to(DEVICE)
        return model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()

    def _categorize_and_reorder_tags(self, tags):
        """对标签进行分类和重排序"""
        categorized = defaultdict(list)
        for tag in tags:
            tag_lower = tag.lower().strip()
            found_category = self.tag_to_category.get(tag_lower, "general")
            categorized[found_category].append(tag)
        
        reordered_tags = []
        for category in TAG_CATEGORIES_ORDER:
            if category in categorized:
                unique_tags = sorted(list(set(categorized[category])), key=lambda x: tags.index(x))
                reordered_tags.extend(unique_tags)
        
        return ", ".join(reordered_tags)
    
    def get_token_counts(self, text):
        """计算Token数量"""
        tokens1 = self.te1_tokenizer(text, truncation=False, add_special_tokens=True)['input_ids']
        tokens2 = self.te2_tokenizer(text, truncation=False, add_special_tokens=True)['input_ids']
        return len(tokens1), len(tokens2)

    def _find_image_file(self, base_name, source_dir):
        """查找对应的图片文件"""
        source_path = Path(source_dir)
        
        # 直接匹配
        for ext in IMAGE_EXTENSIONS:
            candidate = source_path / f"{base_name}{ext}"
            if candidate.exists():
                return candidate
        
        # 大小写不敏感匹配
        all_files = list(source_path.glob("*"))
        base_name_lower = base_name.lower()
        
        for file_path in all_files:
            if file_path.stem.lower() == base_name_lower:
                file_ext = file_path.suffix.lower()
                if file_ext in [ext.lower() for ext in IMAGE_EXTENSIONS]:
                    return file_path
        
        return None

    def _process_image(self, source_image, target_image):
        """处理图片：压缩并转换格式"""
        if target_image.exists():
            return {"status": "已存在", "compressed": False, "original_size": None, "final_size": None}
        
        try:
            with Image.open(source_image) as img:
                original_width, original_height = img.size
                original_size = (original_width, original_height)
                
                needs_compression = (original_width > self.max_resolution or 
                                   original_height > self.max_resolution)
                
                if needs_compression:
                    scale_factor = min(self.max_resolution / original_width, 
                                     self.max_resolution / original_height)
                    new_width = int(original_width * scale_factor)
                    new_height = int(original_height * scale_factor)
                    
                    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    target_png = target_image.with_suffix('.png')
                    img_resized.save(target_png, format='PNG', optimize=True)
                    
                    return {
                        "status": "压缩成功", 
                        "compressed": True,
                        "original_size": original_size,
                        "final_size": (new_width, new_height),
                        "scale_factor": scale_factor,
                        "saved_path": str(target_png)
                    }
                else:
                    target_png = target_image.with_suffix('.png')
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    img.save(target_png, format='PNG', optimize=True)
                    
                    return {
                        "status": "转换成功",
                        "compressed": False,
                        "original_size": original_size,
                        "final_size": original_size,
                        "scale_factor": 1.0,
                        "saved_path": str(target_png)
                    }
                    
        except Exception as e:
            return {
                "status": f"失败: {e}",
                "compressed": False,
                "original_size": None,
                "final_size": None,
                "error": str(e)
            }

    def process_batch(self, batch_files, source_dir, output_dir, batch_id):
        """处理单个批次的文件"""
        print(f"[BATCH-{batch_id}] 开始处理 {len(batch_files)} 个文件")
        
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        batch_results = []
        image_stats = {"found": 0, "missing": 0, "processed": 0, "errors": 0, "compressed": 0, "total_size_reduction": 0}
        
        for txt_file_path in tqdm(batch_files, desc=f"批次-{batch_id}", leave=False):
            try:
                txt_file = Path(txt_file_path)
                
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if not lines or not lines[0].strip():
                    continue

                original_tags_str = lines[0].strip()
                description = lines[1].strip() if len(lines) > 1 else ""
                original_tags_list = [t.strip() for t in original_tags_str.split(',') if t.strip()]

                # 重排序标签
                reordered_tags_str = self._categorize_and_reorder_tags(original_tags_list)
                
                # 计算语义相似度
                original_emb1 = self._get_embedding(original_tags_str, "te1")
                reordered_emb1 = self._get_embedding(reordered_tags_str, "te1")
                sim1 = cosine_similarity(original_emb1, reordered_emb1)[0][0]
                
                original_emb2 = self._get_embedding(original_tags_str, "te2")
                reordered_emb2 = self._get_embedding(reordered_tags_str, "te2")
                sim2 = cosine_similarity(original_emb2, reordered_emb2)[0][0]

                # 计算Token数量
                tokens1, tokens2 = self.get_token_counts(reordered_tags_str)

                # 保存处理后的文本文件
                output_txt_file = Path(output_dir) / txt_file.name
                with open(output_txt_file, 'w', encoding='utf-8') as f:
                    if description:
                        combined_content = f"{reordered_tags_str}, {description}"
                    else:
                        combined_content = reordered_tags_str
                    f.write(combined_content)

                # 处理图片文件
                base_name = txt_file.stem
                source_image = self._find_image_file(base_name, source_dir)
                
                process_result = None
                if source_image:
                    target_image = Path(output_dir) / source_image.name
                    process_result = self._process_image(source_image, target_image)
                    image_stats["found"] += 1
                    
                    if "成功" in process_result["status"]:
                        image_stats["processed"] += 1
                        if process_result["compressed"]:
                            image_stats["compressed"] += 1
                            original_pixels = process_result["original_size"][0] * process_result["original_size"][1]
                            final_pixels = process_result["final_size"][0] * process_result["final_size"][1]
                            size_reduction = (original_pixels - final_pixels) / original_pixels
                            image_stats["total_size_reduction"] += size_reduction
                    elif "失败" in process_result["status"]:
                        image_stats["errors"] += 1
                else:
                    image_stats["missing"] += 1

                # 构建结果记录
                result_entry = {
                    "filename": txt_file.name,
                    "similarity_te1": float(sim1),
                    "similarity_te2": float(sim2),
                    "tokens_te1": tokens1,
                    "tokens_te2": tokens2,
                    "original_length": len(original_tags_str),
                    "reordered_length": len(reordered_tags_str),
                    "image_found": source_image is not None,
                    "image_status": process_result["status"] if source_image else "未找到"
                }
                
                if process_result and source_image:
                    result_entry.update({
                        "image_compressed": process_result.get("compressed", False),
                        "original_resolution": process_result.get("original_size"),
                        "final_resolution": process_result.get("final_size"),
                        "compression_ratio": process_result.get("scale_factor", 1.0)
                    })
                
                batch_results.append(result_entry)

            except Exception as e:
                print(f"[BATCH-{batch_id} ERROR] 处理文件 {txt_file_path} 失败: {e}")

        # 保存批次结果
        batch_report = {
            "batch_id": batch_id,
            "processed_files": len(batch_results),
            "image_stats": image_stats,
            "results": batch_results
        }
        
        batch_output_file = Path(output_dir) / f"batch_{batch_id}_results.json"
        with open(batch_output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, indent=2, ensure_ascii=False)
        
        print(f"[BATCH-{batch_id}] 完成，已处理 {len(batch_results)} 个文件")
        return batch_report


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批次处理器 - 处理单个批次的标签文件")
    parser.add_argument("--batch-files", required=True,
                       help="批次文件列表，逗号分隔")
    parser.add_argument("--source-dir", required=True,
                       help="源数据目录")
    parser.add_argument("--output-dir", required=True,
                       help="输出目录")
    parser.add_argument("--batch-id", type=int, required=True,
                       help="批次ID")
    parser.add_argument("--max-resolution", type=int, default=5000,
                       help="图片最大分辨率")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # 解析批次文件列表
    batch_files = [f.strip() for f in args.batch_files.split(',') if f.strip()]
    
    print(f"[BATCH-{args.batch_id}] 启动批次处理器")
    print(f"[BATCH-{args.batch_id}] 文件数量: {len(batch_files)}")
    print(f"[BATCH-{args.batch_id}] 源目录: {args.source_dir}")
    print(f"[BATCH-{args.batch_id}] 输出目录: {args.output_dir}")
    
    # 创建处理器并执行
    processor = BatchProcessor(max_resolution=args.max_resolution)
    result = processor.process_batch(batch_files, args.source_dir, args.output_dir, args.batch_id)
    
    print(f"[BATCH-{args.batch_id}] 批次处理完成") 