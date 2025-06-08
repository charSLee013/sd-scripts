#!/usr/bin/env python3
"""
触发词效果测试脚本
验证文档中关于触发词和--shuffle_caption的理论
"""

import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_trigger_word_theory():
    """测试触发词理论"""
    print("触发词理论验证测试")
    print("="*50)
    
    # 使用TE1来测试（根据之前结果，它对标签更敏感）
    model_path = "/root/text_encoder/CLIP-ViT-B-32-laion2B-s34B-b79K"
    
    tokenizer = CLIPTokenizer.from_pretrained(model_path)
    model = CLIPTextModel.from_pretrained(model_path).to(device)
    model.eval()
    
    print(f"使用模型: TE1 (ViT-B-32)")
    
    def encode_text(text):
        with torch.no_grad():
            tokens = tokenizer(text, truncation=True, padding="max_length", 
                             max_length=77, return_tensors="pt").to(device)
            outputs = model(**tokens)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    # 测试不同的触发词候选
    print("\n1. 测试触发词独特性")
    print("-" * 30)
    
    trigger_candidates = [
        "ohwx",         # 文档推荐的稀有词
        "zksk",         # 另一个稀有词
        "beautiful",    # 常见词（应该避免）
        "girl",         # 很常见的词（应该避免）
        "MyStyle",      # 可能被分解的词
    ]
    
    # 基础场景描述
    base_scene = "1girl, solo, standing, outdoor, daylight, realistic"
    
    # 测试每个触发词与基础场景的嵌入
    print("触发词独特性测试:")
    for trigger in trigger_candidates:
        # 测试单独的触发词
        trigger_embedding = encode_text(trigger)
        
        # 测试带触发词的完整提示
        full_prompt = f"{trigger}, {base_scene}"
        full_embedding = encode_text(full_prompt)
        
        # 测试基础场景
        base_embedding = encode_text(base_scene)
        
        # 计算相似度
        trigger_base_sim = cosine_similarity(trigger_embedding, base_embedding)[0][0]
        full_base_sim = cosine_similarity(full_embedding, base_embedding)[0][0]
        
        # 分词测试
        tokens = tokenizer(trigger)['input_ids']
        effective_tokens = [t for t in tokens if t not in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]]
        
        print(f"  {trigger:12} | token数: {len(effective_tokens)} | 与基础相似度: {trigger_base_sim:.4f} | 加入后相似度: {full_base_sim:.4f}")
    
    # 测试shuffle_caption的模拟效果
    print(f"\n2. 模拟shuffle_caption效果")
    print("-" * 30)
    
    # 使用一个稀有触发词
    trigger = "ohwx"
    
    # 原始固定顺序的标题
    fixed_caption = f"{trigger}, 1girl, brown hair, blue eyes, smiling, standing"
    
    # 模拟打乱后的不同组合（保持触发词在前）
    shuffled_captions = [
        f"{trigger}, blue eyes, 1girl, standing, smiling, brown hair",
        f"{trigger}, smiling, brown hair, 1girl, blue eyes, standing", 
        f"{trigger}, standing, blue eyes, smiling, 1girl, brown hair",
        f"{trigger}, 1girl, standing, brown hair, smiling, blue eyes"
    ]
    
    print("固定顺序vs打乱顺序的一致性测试:")
    
    # 编码固定顺序
    fixed_embedding = encode_text(fixed_caption)
    
    # 编码所有打乱版本并计算与固定版本的相似度
    similarities = []
    for i, shuffled in enumerate(shuffled_captions):
        shuffled_embedding = encode_text(shuffled)
        sim = cosine_similarity(fixed_embedding, shuffled_embedding)[0][0]
        similarities.append(sim)
        print(f"  变体 {i+1}: {sim:.4f}")
    
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    
    print(f"\n  平均相似度: {avg_similarity:.4f} ± {std_similarity:.4f}")
    
    if avg_similarity > 0.95:
        print("  ✓ 高相似度说明触发词能保持概念一致性")
    else:
        print("  ? 相似度偏低，可能需要进一步调查")
    
    # 测试结构化格式的效果
    print(f"\n3. 测试结构化格式效果")
    print("-" * 30)
    
    # 加载一个实际的数据样本
    data_path = Path("/root/data/cluster_4")
    txt_files = list(data_path.glob("*.txt"))
    
    if txt_files:
        with open(txt_files[0], 'r', encoding='utf-8') as f:
            content = f.read().strip()
            lines = content.split('\n')
            
            if len(lines) >= 2:
                original_tags = lines[0].strip()
                original_desc = lines[1].strip()
                
                # 不同的格式组合
                formats = {
                    "分离-标签": original_tags,
                    "分离-描述": original_desc,
                    "简单拼接": f"{original_tags}, {original_desc}",
                    "结构化拼接": f"tags: {original_tags}. description: {original_desc}",
                    "带触发词": f"ohwx, {original_tags}, {original_desc}"
                }
                
                print("不同格式的嵌入对比:")
                embeddings = {}
                for name, text in formats.items():
                    embeddings[name] = encode_text(text)
                    print(f"  {name}: {len(tokenizer(text)['input_ids'])} tokens")
                
                print(f"\n格式间相似度矩阵:")
                format_names = list(formats.keys())
                for i, name1 in enumerate(format_names):
                    for j, name2 in enumerate(format_names):
                        if j <= i:
                            continue
                        sim = cosine_similarity(embeddings[name1], embeddings[name2])[0][0]
                        print(f"  {name1} vs {name2}: {sim:.4f}")

def main():
    test_trigger_word_theory()

if __name__ == "__main__":
    main() 