#!/usr/bin/env python3
"""
简化的TE1和TE2测试脚本
验证文档中关于两个text encoder对不同文本格式响应差异的理论
"""

import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
import random
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def load_and_test_model(model_path, model_name, sample_data):
    """加载模型并测试"""
    print(f"\n{'='*50}")
    print(f"测试模型: {model_name}")
    print(f"路径: {model_path}")
    print(f"{'='*50}")
    
    try:
        # 加载模型
        tokenizer = CLIPTokenizer.from_pretrained(model_path)
        model = CLIPTextModel.from_pretrained(model_path).to(device)
        model.eval()
        
        print(f"✓ 模型加载成功")
        print(f"词汇表大小: {tokenizer.vocab_size}")
        print(f"最大token长度: {tokenizer.model_max_length}")
        
        results = []
        
        for i, (tags, description, filename) in enumerate(sample_data):
            print(f"\n--- 样本 {i+1}: {filename} ---")
            
            with torch.no_grad():
                # 处理标签文本
                tags_tokens = tokenizer(
                    tags, truncation=True, padding="max_length", 
                    max_length=77, return_tensors="pt"
                ).to(device)
                
                tags_output = model(**tags_tokens)
                tags_embedding = tags_output.last_hidden_state.mean(dim=1)
                
                # 处理描述文本
                desc_tokens = tokenizer(
                    description, truncation=True, padding="max_length", 
                    max_length=77, return_tensors="pt"
                ).to(device)
                
                desc_output = model(**desc_tokens)
                desc_embedding = desc_output.last_hidden_state.mean(dim=1)
                
                # 计算相似度
                similarity = cosine_similarity(
                    tags_embedding.cpu().numpy(),
                    desc_embedding.cpu().numpy()
                )[0][0]
                
                # 计算有效token数
                tags_valid_tokens = (tags_tokens['input_ids'] != tokenizer.pad_token_id).sum().item()
                desc_valid_tokens = (desc_tokens['input_ids'] != tokenizer.pad_token_id).sum().item()
                
                print(f"标签文本长度: {tags_valid_tokens} tokens")
                print(f"描述文本长度: {desc_valid_tokens} tokens") 
                print(f"语义相似度: {similarity:.4f}")
                
                # 显示部分token
                tags_token_list = tokenizer.convert_ids_to_tokens(tags_tokens['input_ids'][0])
                desc_token_list = tokenizer.convert_ids_to_tokens(desc_tokens['input_ids'][0])
                
                print(f"标签前5个token: {tags_token_list[:5]}")
                print(f"描述前5个token: {desc_token_list[:5]}")
                
                results.append({
                    'sample': filename,
                    'tags_tokens': tags_valid_tokens,
                    'desc_tokens': desc_valid_tokens,
                    'similarity': float(similarity),
                    'tags_preview': tags[:50] + "..." if len(tags) > 50 else tags,
                    'desc_preview': description[:50] + "..." if len(description) > 50 else description
                })
        
        return results
        
    except Exception as e:
        print(f"✗ 测试模型 {model_name} 失败: {e}")
        return []

def main():
    """主函数"""
    print("SDXL双Text Encoder性能对比测试")
    print("验证文档中关于TE1和TE2功能解耦的理论")
    
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    
    # 模型路径配置
    models = {
        "TE1_ViT-B-32": "/root/text_encoder/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "TE2_ViT-L-14": "/root/text_encoder/clip-vit-large-patch14"
    }
    
    # 加载数据样本
    print("\n正在加载数据集样本...")
    data_path = Path("/root/data/cluster_4")
    txt_files = list(data_path.glob("*.txt"))
    
    # 随机选择3个样本
    selected_files = random.sample(txt_files, min(3, len(txt_files)))
    
    sample_data = []
    for txt_file in selected_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                lines = content.split('\n')
                
                if len(lines) >= 2:
                    tags = lines[0].strip()
                    description = lines[1].strip()
                    sample_data.append((tags, description, txt_file.name))
        except Exception as e:
            print(f"读取文件 {txt_file} 失败: {e}")
    
    print(f"成功加载 {len(sample_data)} 个样本")
    
    # 测试每个模型
    all_results = {}
    for model_name, model_path in models.items():
        results = load_and_test_model(model_path, model_name, sample_data)
        all_results[model_name] = results
    
    # 对比分析
    print(f"\n{'='*60}")
    print("对比分析结果")
    print(f"{'='*60}")
    
    if 'TE1_ViT-B-32' in all_results and 'TE2_ViT-L-14' in all_results:
        te1_results = all_results['TE1_ViT-B-32']
        te2_results = all_results['TE2_ViT-L-14']
        
        if te1_results and te2_results:
            te1_similarities = [r['similarity'] for r in te1_results]
            te2_similarities = [r['similarity'] for r in te2_results]
            
            print(f"TE1 (ViT-B-32) 相似度:")
            print(f"  平均值: {np.mean(te1_similarities):.4f}")
            print(f"  标准差: {np.std(te1_similarities):.4f}")
            print(f"  范围: {np.min(te1_similarities):.4f} - {np.max(te1_similarities):.4f}")
            
            print(f"\nTE2 (ViT-L-14) 相似度:")
            print(f"  平均值: {np.mean(te2_similarities):.4f}")
            print(f"  标准差: {np.std(te2_similarities):.4f}")
            print(f"  范围: {np.min(te2_similarities):.4f} - {np.max(te2_similarities):.4f}")
            
            # 理论验证
            print(f"\n{'='*60}")
            print("理论验证")
            print(f"{'='*60}")
            
            print("文档理论回顾:")
            print("- TE1 (ViT-B-32): 更擅长识别和提取原子化标签")
            print("- TE2 (ViT-L-14): 更擅长理解整体场景和语义关系")
            print("- 双编码器应该对相同输入产生不同的功能关注点")
            
            print(f"\n实验结果:")
            if abs(np.mean(te1_similarities) - np.mean(te2_similarities)) > 0.05:
                print("✓ 两个编码器显示出明显的相似度差异模式")
                print("  这支持了文档中关于功能解耦的理论")
            else:
                print("? 两个编码器的相似度模式相近")
                print("  可能需要更多样本或不同的测试方法")
            
            # 详细对比
            print(f"\n样本级对比:")
            for i in range(min(len(te1_results), len(te2_results))):
                te1_sim = te1_results[i]['similarity']
                te2_sim = te2_results[i]['similarity']
                diff = abs(te1_sim - te2_sim)
                print(f"样本 {i+1}: TE1={te1_sim:.4f}, TE2={te2_sim:.4f}, 差异={diff:.4f}")

if __name__ == "__main__":
    main() 