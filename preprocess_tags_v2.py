import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import CLIPTokenizer, CLIPTextModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# --- 全局配置 ---
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/root/data/cluster_4"
OUTPUT_DIR = "/root/data/cluster_4_restructured_v2"
ANALYSIS_FILE = "tag_analysis_results.json"
TE1_PATH = "/root/text_encoder/clip-vit-large-patch14"
TE2_PATH = "/root/text_encoder/CLIP-ViT-B-32-laion2B-s34B-b79K"

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

class TagRestructuringV2:
    """
    一个集成了语义重排序、精确Token计算和语义保真度验证的
    高级数据重构工具。
    """
    def __init__(self):
        print("[INFO] 正在初始化数据重构工具 V2.0...")
        self._load_models()
        self._build_tag_mapping()

    def _load_models(self):
        """加载TE1和TE2的模型与分词器"""
        print(f"[INFO] 使用设备: {DEVICE}")
        self.te1_tokenizer = CLIPTokenizer.from_pretrained(TE1_PATH)
        self.te1_model = CLIPTextModel.from_pretrained(TE1_PATH).to(DEVICE).eval()
        self.te2_tokenizer = CLIPTokenizer.from_pretrained(TE2_PATH)
        self.te2_model = CLIPTextModel.from_pretrained(TE2_PATH).to(DEVICE).eval()
        print("[INFO] TE1 和 TE2 模型加载完毕。")

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
        """对标签进行分类和重排序，不添加额外标记"""
        categorized = defaultdict(list)
        for tag in tags:
            tag_lower = tag.lower().strip()
            found_category = self.tag_to_category.get(tag_lower, "general")
            categorized[found_category].append(tag)
        
        reordered_tags = []
        for category in TAG_CATEGORIES_ORDER:
            if category in categorized:
                # 保持原始大小写，去重
                unique_tags = sorted(list(set(categorized[category])), key=lambda x: tags.index(x))
                reordered_tags.extend(unique_tags)
        
        return ", ".join(reordered_tags)
    
    def get_token_counts(self, text):
        """使用真实分词器计算Token数量"""
        tokens1 = self.te1_tokenizer(text, truncation=False, add_special_tokens=True)['input_ids']
        tokens2 = self.te2_tokenizer(text, truncation=False, add_special_tokens=True)['input_ids']
        return len(tokens1), len(tokens2)

    def run(self):
        """执行完整的数据重构和验证流程"""
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"[INFO] 已创建输出目录: {OUTPUT_DIR}")

        all_txt_files = list(Path(DATA_DIR).glob("*.txt"))
        print(f"[INFO] 发现 {len(all_txt_files)} 个标签文件需要处理。")

        report_data = []
        pbar = tqdm(all_txt_files, desc="[V2.0] 数据重构与验证")
        for txt_file in pbar:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if not lines or not lines[0].strip():
                    continue

                original_tags_str = lines[0].strip()
                description = lines[1].strip() if len(lines) > 1 else ""
                original_tags_list = [t.strip() for t in original_tags_str.split(',') if t.strip()]

                # 1. 重排序
                reordered_tags_str = self._categorize_and_reorder_tags(original_tags_list)
                
                # 2. 计算语义相似度
                original_emb1 = self._get_embedding(original_tags_str, "te1")
                reordered_emb1 = self._get_embedding(reordered_tags_str, "te1")
                sim1 = cosine_similarity(original_emb1, reordered_emb1)[0][0]
                
                original_emb2 = self._get_embedding(original_tags_str, "te2")
                reordered_emb2 = self._get_embedding(reordered_tags_str, "te2")
                sim2 = cosine_similarity(original_emb2, reordered_emb2)[0][0]

                # 3. 计算精确Token长度
                tokens1, tokens2 = self.get_token_counts(reordered_tags_str)

                # 4. 保存重构文件
                with open(Path(OUTPUT_DIR) / txt_file.name, 'w', encoding='utf-8') as f:
                    f.write(reordered_tags_str + '\n')
                    if description:
                        f.write(description + '\n')

                report_data.append({
                    "filename": txt_file.name,
                    "similarity_te1": float(sim1),
                    "similarity_te2": float(sim2),
                    "tokens_te1": tokens1,
                    "tokens_te2": tokens2,
                    "original_length": len(original_tags_str),
                    "reordered_length": len(reordered_tags_str)
                })

            except Exception as e:
                print(f"[ERROR] 处理文件 {txt_file.name} 失败: {e}")

        # 5. 生成报告
        self._generate_report(report_data)

    def _generate_report(self, report_data):
        """生成最终的量化分析报告"""
        print("\n" + "="*80)
        print("📊 SDXL数据重构V2.0 - 最终量化报告")
        print("="*80)

        if not report_data:
            print("[ERROR] 没有处理任何文件，无法生成报告。")
            return

        avg_sim1 = np.mean([d['similarity_te1'] for d in report_data])
        avg_sim2 = np.mean([d['similarity_te2'] for d in report_data])
        avg_tokens1 = np.mean([d['tokens_te1'] for d in report_data])
        avg_tokens2 = np.mean([d['tokens_te2'] for d in report_data])
        max_tokens1 = max([d['tokens_te1'] for d in report_data])
        max_tokens2 = max([d['tokens_te2'] for d in report_data])

        print(f"      处理文件总数: {len(report_data)}")
        print("\n--- 语义保真度 (越高越好) ---")
        print(f"      TE1 平均相似度: {avg_sim1:.4f}")
        print(f"      TE2 平均相似度: {avg_sim2:.4f}")
        print(f"      综合平均相似度: {(avg_sim1 + avg_sim2) / 2:.4f}")
        
        print("\n--- Token长度分析 (越低越好) ---")
        print(f"      TE1 平均Token数: {avg_tokens1:.1f} (最大: {max_tokens1})")
        print(f"      TE2 平均Token数: {avg_tokens2:.1f} (最大: {max_tokens2})")

        # 将报告保存到JSON文件
        report_path = "data_reconstruction_report_v2.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "file_count": len(report_data),
                    "avg_similarity_te1": avg_sim1,
                    "avg_similarity_te2": avg_sim2,
                    "avg_tokens_te1": avg_tokens1,
                    "max_tokens_te1": max_tokens1,
                    "avg_tokens_te2": avg_tokens2,
                    "max_tokens_te2": max_tokens2
                },
                "details": report_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] V2.0数据重构完成！详细报告已保存至: {report_path}")
        print("="*80)


if __name__ == "__main__":
    restructurer = TagRestructuringV2()
    restructurer.run() 