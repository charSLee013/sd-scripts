#!/usr/bin/env python3
"""
本地CLIP文本编码器测试脚本
测试TE1和TE2对不同文本格式的响应差异
"""

import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
import random
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import json

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class LocalCLIPAnalyzer:
    def __init__(self, model_path, model_name):
        """初始化本地CLIP文本分析器"""
        self.model_name = model_name
        self.model_path = model_path
        print(f"正在加载本地模型: {model_name} from {model_path}")
        
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
            self.model = CLIPTextModel.from_pretrained(model_path).to(device)
            self.model.eval()
            print(f"模型 {model_name} 加载成功")
            
            # 获取模型配置信息
            print(f"词汇表大小: {self.tokenizer.vocab_size}")
            print(f"最大长度: {self.tokenizer.model_max_length}")
            
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {e}")
            raise
    
    def tokenize_and_analyze(self, text, max_length=77):
        """分词并分析文本"""
        with torch.no_grad():
            # 分词
            tokens = self.tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # 获取文本embedding
            outputs = self.model(**tokens)
            
            # 使用最后一层hidden states的平均作为文本表示
            pooled_output = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]
            
            # 计算有效token数量（排除padding）
            attention_mask = tokens['attention_mask']
            valid_tokens = attention_mask.sum().item()
            
            return {
                'embedding': pooled_output.cpu().numpy(),
                'token_count': valid_tokens,
                'tokens': self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]),
                'input_ids': tokens['input_ids'][0].cpu().numpy(),
                'attention_mask': attention_mask[0].cpu().numpy()
            }
    
    def compare_texts(self, tags_text, description_text):
        """比较两种文本类型"""
        
        print(f"\n--- {self.model_name} 分析 ---")
        print(f"标签文本: {tags_text[:100]}...")
        print(f"描述文本: {description_text[:100]}...")
        
        # 分析两种文本
        tags_result = self.tokenize_and_analyze(tags_text)
        desc_result = self.tokenize_and_analyze(description_text)
        
        # 计算相似度
        similarity = cosine_similarity(
            tags_result['embedding'], 
            desc_result['embedding']
        )[0][0]
        
        print(f"标签Token数: {tags_result['token_count']}")
        print(f"描述Token数: {desc_result['token_count']}")
        print(f"语义相似度: {similarity:.4f}")
        
        # 显示前10个token
        print(f"标签前10个token: {tags_result['tokens'][:10]}")
        print(f"描述前10个token: {desc_result['tokens'][:10]}")
        
        return {
            'model': self.model_name,
            'tags_tokens': tags_result['token_count'],
            'desc_tokens': desc_result['token_count'],
            'similarity': similarity,
            'tags_embedding': tags_result['embedding'],
            'desc_embedding': desc_result['embedding']
        }

def load_sample_data(data_dir, num_samples=5):
    """加载样本数据"""
    data_path = Path(data_dir)
    txt_files = list(data_path.glob("*.txt"))
    
    if len(txt_files) < num_samples:
        num_samples = len(txt_files)
    
    # 随机选择样本
    selected_files = random.sample(txt_files, num_samples)
    
    samples = []
    for txt_file in selected_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                lines = content.split('\n')
                
                if len(lines) >= 2:
                    tags = lines[0].strip()
                    description = lines[1].strip()
                    
                    samples.append({
                        'file': txt_file.name,
                        'tags': tags,
                        'description': description
                    })
        except Exception as e:
            print(f"读取文件 {txt_file} 失败: {e}")
    
    return samples

def test_structured_format(analyzer, tags, description):
    """测试结构化格式"""
    
    print(f"\n=== 测试结构化格式 ===")
    
    # 原始分离格式
    print("1. 原始分离格式:")
    original_tags = analyzer.tokenize_and_analyze(tags)
    original_desc = analyzer.tokenize_and_analyze(description)
    
    # 简单拼接格式
    simple_concat = f"{tags}, {description}"
    print("2. 简单拼接格式:")
    simple_result = analyzer.tokenize_and_analyze(simple_concat)
    
    # 结构化拼接格式（文档推荐）
    structured_concat = f"tags: {tags}. description: {description}"
    print("3. 结构化拼接格式:")
    structured_result = analyzer.tokenize_and_analyze(structured_concat)
    
    # 计算各种相似度
    tag_desc_sim = cosine_similarity(
        original_tags['embedding'], 
        original_desc['embedding']
    )[0][0]
    
    simple_vs_tags = cosine_similarity(
        simple_result['embedding'], 
        original_tags['embedding']
    )[0][0]
    
    structured_vs_tags = cosine_similarity(
        structured_result['embedding'], 
        original_tags['embedding']
    )[0][0]
    
    simple_vs_desc = cosine_similarity(
        simple_result['embedding'], 
        original_desc['embedding']
    )[0][0]
    
    structured_vs_desc = cosine_similarity(
        structured_result['embedding'], 
        original_desc['embedding']
    )[0][0]
    
    print(f"\n相似度分析:")
    print(f"原始标签 vs 原始描述: {tag_desc_sim:.4f}")
    print(f"简单拼接 vs 原始标签: {simple_vs_tags:.4f}")
    print(f"简单拼接 vs 原始描述: {simple_vs_desc:.4f}")
    print(f"结构化拼接 vs 原始标签: {structured_vs_tags:.4f}")
    print(f"结构化拼接 vs 原始描述: {structured_vs_desc:.4f}")
    
    return {
        'tag_desc_similarity': tag_desc_sim,
        'simple_vs_tags': simple_vs_tags,
        'simple_vs_desc': simple_vs_desc,
        'structured_vs_tags': structured_vs_tags,
        'structured_vs_desc': structured_vs_desc
    }

def main():
    """主函数"""
    print("本地CLIP文本编码器测试")
    print("=" * 50)
    
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    
    # 本地模型路径
    models = {
        "TE1_ViT-B-32": "/root/text_encoder/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "TE2_ViT-L-14": "/root/text_encoder/clip-vit-large-patch14"
    }
    
    # 加载样本数据
    print("正在加载数据集样本...")
    samples = load_sample_data("/root/data/cluster_4", num_samples=3)
    print(f"成功加载 {len(samples)} 个样本")
    
    all_results = []
    
    # 测试每个模型
    for model_name, model_path in models.items():
        try:
            print(f"\n{'='*60}")
            print(f"测试模型: {model_name}")
            print(f"{'='*60}")
            
            analyzer = LocalCLIPAnalyzer(model_path, model_name)
            
            # 对每个样本进行分析
            for i, sample in enumerate(samples):
                print(f"\n--- 样本 {i+1}: {sample['file']} ---")
                
                # 基础比较
                result = analyzer.compare_texts(sample['tags'], sample['description'])
                result['sample_file'] = sample['file']
                result['tags_text'] = sample['tags']
                result['desc_text'] = sample['description']
                
                # 结构化格式测试（只对第一个样本）
                if i == 0:
                    structured_results = test_structured_format(
                        analyzer, sample['tags'], sample['description']
                    )
                    result['structured_analysis'] = structured_results
                
                all_results.append(result)
                
        except Exception as e:
            print(f"测试模型 {model_name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    print(f"\n{'='*60}")
    print("保存测试结果...")
    
    output_file = "/root/local_clip_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # 处理numpy数组以便JSON序列化
        json_results = []
        for result in all_results:
            json_result = result.copy()
            if 'tags_embedding' in json_result:
                json_result['tags_embedding'] = result['tags_embedding'].tolist()
            if 'desc_embedding' in json_result:
                json_result['desc_embedding'] = result['desc_embedding'].tolist()
            # 确保所有numpy类型都转换为Python原生类型
            if 'similarity' in json_result:
                json_result['similarity'] = float(json_result['similarity'])
            if 'structured_analysis' in json_result:
                for key, value in json_result['structured_analysis'].items():
                    if isinstance(value, np.floating):
                        json_result['structured_analysis'][key] = float(value)
            json_results.append(json_result)
        
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_file}")
    
    # 总结分析
    print(f"\n{'='*60}")
    print("总结分析")
    print(f"{'='*60}")
    
    # 按模型分组
    te1_results = [r for r in all_results if 'TE1' in r['model']]
    te2_results = [r for r in all_results if 'TE2' in r['model']]
    
    if te1_results and te2_results:
        te1_similarities = [r['similarity'] for r in te1_results]
        te2_similarities = [r['similarity'] for r in te2_results]
        
        print(f"TE1 (ViT-B-32) 平均相似度: {np.mean(te1_similarities):.4f} ± {np.std(te1_similarities):.4f}")
        print(f"TE2 (ViT-L-14) 平均相似度: {np.mean(te2_similarities):.4f} ± {np.std(te2_similarities):.4f}")
        
        print(f"\n这验证了文档中的理论吗？")
        print("- 如果TE1和TE2显示不同的相似度模式，说明它们确实对文本有不同的理解")
        print("- 较大的相似度差异可能表明一个模型更适合某种文本格式")

if __name__ == "__main__":
    main() 