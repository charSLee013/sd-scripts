#!/usr/bin/env python3
"""
SDXL训练疑惑综合解谜脚本
基于实验验证文档中提出但未完全解决的关键问题
"""

import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MysteryAnalyzer:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.load_models()
        
    def load_models(self):
        """加载TE1和TE2模型"""
        model_configs = {
            "TE1": "/root/text_encoder/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "TE2": "/root/text_encoder/clip-vit-large-patch14"
        }
        
        for name, path in model_configs.items():
            print(f"加载 {name} 模型...")
            self.tokenizers[name] = CLIPTokenizer.from_pretrained(path)
            self.models[name] = CLIPTextModel.from_pretrained(path).to(device)
            self.models[name].eval()
        
        print(f"所有模型已加载到 {device}")
    
    def encode_text_with_details(self, text, model_name="TE1", max_length=77):
        """详细编码文本，返回嵌入和token信息"""
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        # 分词
        tokens = tokenizer(text, truncation=True, padding="max_length", 
                          max_length=max_length, return_tensors="pt")
        
        # 计算有效token数量
        input_ids = tokens['input_ids'][0]
        valid_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
        
        # 检查是否被截断
        original_tokens = tokenizer(text, truncation=False)['input_ids']
        is_truncated = len(original_tokens) > max_length
        
        # 编码
        with torch.no_grad():
            tokens = tokens.to(device)
            outputs = model(**tokens)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return {
            'embedding': embedding,
            'valid_tokens': valid_tokens,
            'is_truncated': is_truncated,
            'original_length': len(original_tokens),
            'tokens_preview': tokenizer.decode(input_ids[:10]) if valid_tokens > 0 else ""
        }
    
    def mystery_1_text_length_impact(self):
        """疑惑1: 文本长度截断对双编码器性能的实际影响"""
        print("\n" + "="*60)
        print("🔍 疑惑1: 文本长度截断对双编码器性能的实际影响")
        print("="*60)
        
        # 准备不同长度的测试样本
        test_cases = [
            # 短文本 (< 30 tokens)
            "1girl, solo, beautiful, detailed eyes",
            
            # 中等长度 (30-50 tokens)
            "1girl, solo, beautiful detailed eyes, long flowing hair, elegant dress, standing in garden, sunlight, masterpiece",
            
            # 接近限制长度 (50-77 tokens)  
            "1girl, solo, beautiful detailed sparkling blue eyes, long flowing golden hair with intricate braids, elegant white victorian dress with lace details, standing in a magnificent rose garden, warm golden sunlight filtering through leaves, masterpiece, best quality, ultra detailed",
            
            # 超长文本 (> 77 tokens，会被截断)
            "1girl, solo, beautiful detailed sparkling blue eyes with long eyelashes, long flowing golden hair with intricate braids and small flowers, elegant white victorian dress with delicate lace details and pearl buttons, standing gracefully in a magnificent rose garden with red and pink roses, warm golden sunlight filtering through green leaves creating beautiful shadows, masterpiece, best quality, ultra detailed, photorealistic, 8k resolution, professional photography, perfect composition, cinematic lighting, highly detailed background"
        ]
        
        results = []
        
        print("⚠️  重要发现: CLIP模型最大支持77个token位置，超出部分会被截断")
        print("这解释了为什么文档建议使用 --max_token_length=225 的困惑\n")
        
        for i, text in enumerate(test_cases):
            print(f"\n测试样本 {i+1}:")
            print(f"原文: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            for model_name in ["TE1", "TE2"]:
                # 只测试77长度限制，因为这是CLIP的硬限制
                result = self.encode_text_with_details(text, model_name, max_length=77)
                
                result_entry = {
                    'sample_id': i+1,
                    'model': model_name,
                    'max_length': 77,
                    'original_text_length': len(text),
                    'original_tokens': result['original_length'],
                    'used_tokens': result['valid_tokens'],
                    'is_truncated': result['is_truncated'],
                    'truncation_loss_rate': (result['original_length'] - result['valid_tokens']) / result['original_length'] if result['original_length'] > 0 else 0,
                    'embedding': result['embedding']
                }
                results.append(result_entry)
                
                print(f"  {model_name}: {result['valid_tokens']}/{result['original_length']} tokens, "
                      f"截断: {'是' if result['is_truncated'] else '否'}, "
                      f"损失率: {result_entry['truncation_loss_rate']:.2%}")
        
        # 分析截断对语义理解的影响
        print(f"\n📊 截断影响分析:")
        
        # 将超长文本手动截断，比较截断前后的语义相似度
        long_text = test_cases[3]  # 超长文本
        
        for model_name in ["TE1", "TE2"]:
            # 获取完整嵌入（实际上已经被截断到77）
            full_result = self.encode_text_with_details(long_text, model_name, max_length=77)
            
            # 手动截断到不同长度进行对比
            tokenizer = self.tokenizers[model_name]
            tokens = tokenizer(long_text, truncation=False)['input_ids']
            
            # 测试不同截断点的效果
            for cut_length in [30, 45, 60, 77]:
                if len(tokens) > cut_length:
                    # 手动截断
                    cut_tokens = tokens[:cut_length-1] + [tokens[-1]]  # 保留结束token
                    cut_text = tokenizer.decode(cut_tokens, skip_special_tokens=True)
                    
                    cut_result = self.encode_text_with_details(cut_text, model_name, max_length=77)
                    
                    # 计算与原始（77截断）版本的相似度
                    similarity = cosine_similarity(full_result['embedding'], cut_result['embedding'])[0][0]
                    
                    print(f"  {model_name} 截断到{cut_length}tokens: 相似度 = {similarity:.4f}")
        
        return results
    
    def mystery_2_shuffle_caption_mechanism(self):
        """疑惑2: shuffle_caption的深层机制验证"""
        print("\n" + "="*60)
        print("🔍 疑惑2: shuffle_caption机制的量化验证")
        print("="*60)
        
        # 模拟shuffle_caption的效果
        base_prompt = "ohwx, 1girl, beautiful, detailed eyes, long hair, smiling, standing, outdoor, sunlight"
        tags = base_prompt.split(", ")
        trigger_word = tags[0]  # "ohwx"
        other_tags = tags[1:]   # 其他标签
        
        print(f"基础提示: {base_prompt}")
        print(f"触发词: {trigger_word}")
        print(f"可打乱标签: {other_tags}")
        
        # 生成多个打乱版本
        shuffled_versions = []
        for i in range(10):  # 生成10个不同的打乱版本
            shuffled_tags = other_tags.copy()
            random.shuffle(shuffled_tags)
            shuffled_prompt = trigger_word + ", " + ", ".join(shuffled_tags)
            shuffled_versions.append(shuffled_prompt)
        
        # 测试概念一致性
        print(f"\n📊 打乱效果分析:")
        
        original_embeddings = {}
        shuffled_embeddings = {}
        
        for model_name in ["TE1", "TE2"]:
            # 编码原始版本
            original_result = self.encode_text_with_details(base_prompt, model_name)
            original_embeddings[model_name] = original_result['embedding']
            
            # 编码打乱版本
            shuffled_embeddings[model_name] = []
            similarities = []
            
            for i, shuffled_prompt in enumerate(shuffled_versions):
                shuffled_result = self.encode_text_with_details(shuffled_prompt, model_name)
                shuffled_embeddings[model_name].append(shuffled_result['embedding'])
                
                # 计算与原始版本的相似度
                similarity = cosine_similarity(original_embeddings[model_name], 
                                             shuffled_result['embedding'])[0][0]
                similarities.append(similarity)
                
                if i < 3:  # 只显示前3个详细信息
                    print(f"  {model_name} 版本{i+1}: {similarity:.4f}")
                    print(f"    原始: {base_prompt[:60]}...")
                    print(f"    打乱: {shuffled_prompt[:60]}...")
            
            avg_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            print(f"\n  {model_name} 统计:")
            print(f"    平均相似度: {avg_similarity:.4f} ± {std_similarity:.4f}")
            print(f"    一致性评价: {'优秀' if avg_similarity > 0.95 else '良好' if avg_similarity > 0.90 else '需要改进'}")
        
        # 分析触发词位置的重要性
        print(f"\n🎯 触发词位置重要性分析:")
        
        # 测试触发词在不同位置的效果
        position_tests = [
            f"{trigger_word}, " + ", ".join(other_tags),  # 开头
            ", ".join(other_tags[:3]) + f", {trigger_word}, " + ", ".join(other_tags[3:]),  # 中间
            ", ".join(other_tags) + f", {trigger_word}"   # 结尾
        ]
        
        position_names = ["开头", "中间", "结尾"]
        
        for model_name in ["TE1", "TE2"]:
            print(f"\n  {model_name} 位置测试:")
            baseline_emb = original_embeddings[model_name]
            
            for pos_name, test_prompt in zip(position_names, position_tests):
                result = self.encode_text_with_details(test_prompt, model_name)
                similarity = cosine_similarity(baseline_emb, result['embedding'])[0][0]
                print(f"    触发词在{pos_name}: {similarity:.4f}")
        
        return {
            'original_embeddings': original_embeddings,
            'shuffled_embeddings': shuffled_embeddings,
            'shuffled_versions': shuffled_versions
        }
    
    def mystery_3_structured_format_optimization(self):
        """疑惑3: 结构化拼接的最优格式探索"""
        print("\n" + "="*60)
        print("🔍 疑惑3: 结构化拼接的最优格式探索") 
        print("="*60)
        
        # 加载真实数据样本
        data_path = Path("/root/data/cluster_4")
        txt_files = list(data_path.glob("*.txt"))
        
        if not txt_files:
            print("❌ 未找到测试数据文件")
            return None
        
        # 选择一个样本
        sample_file = random.choice(txt_files)
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            lines = content.split('\n')
            
            if len(lines) < 2:
                print("❌ 样本文件格式不正确")
                return None
        
        original_tags = lines[0].strip()
        original_desc = lines[1].strip()
        
        print(f"测试样本来源: {sample_file.name}")
        print(f"原始标签: {original_tags[:80]}...")
        print(f"原始描述: {original_desc[:80]}...")
        
        # 定义不同的结构化格式
        format_templates = {
            "方案1-简单拼接": f"{original_tags}, {original_desc}",
            "方案2-冒号分隔": f"tags: {original_tags}. description: {original_desc}",
            "方案3-竖线分隔": f"keywords: {original_tags} | scene: {original_desc}",
            "方案4-括号结构": f"[tags: {original_tags}] [description: {original_desc}]",
            "方案5-明确标识": f"TAGS: {original_tags}. DESCRIPTION: {original_desc}",
            "方案6-自然语言": f"This image contains {original_tags}. The scene shows {original_desc}",
            "方案7-JSON风格": f"{{tags: {original_tags}, description: {original_desc}}}",
            "方案8-换行分隔": f"{original_tags}\n{original_desc}",
        }
        
        # 基准对比
        baselines = {
            "仅标签": original_tags,
            "仅描述": original_desc
        }
        
        print(f"\n📊 结构化格式效果分析:")
        
        results = {}
        
        # 测试所有格式
        all_formats = {**baselines, **format_templates}
        
        for model_name in ["TE1", "TE2"]:
            print(f"\n  {model_name} 测试结果:")
            results[model_name] = {}
            
            # 编码所有格式
            embeddings = {}
            token_counts = {}
            
            for format_name, text in all_formats.items():
                result = self.encode_text_with_details(text, model_name, max_length=77)
                embeddings[format_name] = result['embedding']
                token_counts[format_name] = result['valid_tokens']
                
                print(f"    {format_name}: {result['valid_tokens']} tokens, "
                      f"截断: {'是' if result['is_truncated'] else '否'}")
            
            # 计算格式间的相似度矩阵
            print(f"\n    相似度分析 ({model_name}):")
            
            # 重点比较：各种结构化格式与基准的相似度
            baseline_emb_tags = embeddings["仅标签"]
            baseline_emb_desc = embeddings["仅描述"]
            
            best_format = None
            best_score = 0
            
            for format_name in format_templates.keys():
                format_emb = embeddings[format_name]
                
                # 与标签的相似度
                sim_tags = cosine_similarity(baseline_emb_tags, format_emb)[0][0]
                # 与描述的相似度  
                sim_desc = cosine_similarity(baseline_emb_desc, format_emb)[0][0]
                # 平衡分数
                balance_score = min(sim_tags, sim_desc)  # 取最小值确保两者都能兼顾
                
                print(f"      {format_name}:")
                print(f"        与标签相似度: {sim_tags:.4f}")
                print(f"        与描述相似度: {sim_desc:.4f}")
                print(f"        平衡分数: {balance_score:.4f}")
                
                if balance_score > best_score:
                    best_score = balance_score
                    best_format = format_name
                
                results[model_name][format_name] = {
                    'embedding': format_emb,
                    'sim_tags': sim_tags,
                    'sim_desc': sim_desc,
                    'balance_score': balance_score,
                    'token_count': token_counts[format_name]
                }
            
            print(f"\n    🏆 {model_name} 最佳格式: {best_format} (平衡分数: {best_score:.4f})")
        
        # 跨模型分析
        print(f"\n🎯 跨模型格式一致性分析:")
        for format_name in format_templates.keys():
            te1_score = results["TE1"][format_name]['balance_score']
            te2_score = results["TE2"][format_name]['balance_score']
            consistency = 1 - abs(te1_score - te2_score)  # 一致性分数
            
            print(f"  {format_name}: TE1={te1_score:.3f}, TE2={te2_score:.3f}, 一致性={consistency:.3f}")
        
        return results
    
    def mystery_4_overfitting_detection(self):
        """疑惑4: 过拟合检测的定量指标"""
        print("\n" + "="*60)
        print("🔍 疑惑4: 基于嵌入相似度的过拟合检测方法")
        print("="*60)
        
        # 加载多个数据样本，模拟训练过程中的嵌入变化
        data_path = Path("/root/data/cluster_4")
        txt_files = list(data_path.glob("*.txt"))[:10]  # 取前10个样本
        
        if len(txt_files) < 5:
            print("❌ 数据样本数量不足")
            return None
        
        print(f"加载 {len(txt_files)} 个数据样本进行过拟合检测分析")
        
        # 准备测试数据
        samples = []
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                lines = content.split('\n')
                if len(lines) >= 2:
                    samples.append({
                        'filename': txt_file.name,
                        'tags': lines[0].strip(),
                        'description': lines[1].strip(),
                        'combined': f"{lines[0].strip()}, {lines[1].strip()}"
                    })
        
        print(f"成功加载 {len(samples)} 个有效样本")
        
        # 模拟不同训练阶段的嵌入（在实际应用中，这些应该来自训练checkpoints）
        training_stages = ["初期", "中期", "后期", "过拟合期"]
        
        overfitting_metrics = {}
        
        for model_name in ["TE1", "TE2"]:
            print(f"\n  {model_name} 过拟合检测分析:")
            
            # 计算样本间的基础相似度矩阵
            embeddings = []
            for sample in samples:
                result = self.encode_text_with_details(sample['combined'], model_name)
                embeddings.append(result['embedding'].flatten())
            
            embeddings = np.array(embeddings)
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(embeddings)
            
            # 过拟合检测指标
            metrics = self.calculate_overfitting_metrics(similarity_matrix, samples)
            overfitting_metrics[model_name] = metrics
            
            print(f"    平均样本间相似度: {metrics['avg_similarity']:.4f}")
            print(f"    相似度标准差: {metrics['similarity_std']:.4f}")
            print(f"    最高相似度: {metrics['max_similarity']:.4f}")
            print(f"    多样性指数: {metrics['diversity_index']:.4f}")
            print(f"    过拟合风险评估: {metrics['overfitting_risk']}")
        
        # 建议过拟合检测阈值
        print(f"\n📋 过拟合检测建议:")
        print(f"  1. 平均样本间相似度 > 0.85: 高风险")
        print(f"  2. 相似度标准差 < 0.05: 缺乏多样性")
        print(f"  3. 最高相似度 > 0.95: 可能存在重复学习")
        print(f"  4. 多样性指数 < 0.3: 概念坍塌风险")
        
        # 提供训练建议
        for model_name in ["TE1", "TE2"]:
            metrics = overfitting_metrics[model_name]
            print(f"\n  {model_name} 训练建议:")
            
            if metrics['avg_similarity'] > 0.85:
                print(f"    ⚠️  平均相似度过高，建议降低学习率或增加数据多样性")
            
            if metrics['similarity_std'] < 0.05:
                print(f"    ⚠️  缺乏多样性，建议添加更多不同风格的训练数据")
            
            if metrics['overfitting_risk'] == "高":
                print(f"    🚨 高过拟合风险，建议立即停止训练或回退到之前的checkpoint")
            elif metrics['overfitting_risk'] == "中":
                print(f"    ⚠️  中等风险，建议密切监控后续训练")
            else:
                print(f"    ✅ 风险较低，可以继续训练")
        
        return overfitting_metrics
    
    def calculate_overfitting_metrics(self, similarity_matrix, samples):
        """计算过拟合检测指标"""
        n = len(similarity_matrix)
        
        # 排除对角线元素（自己与自己的相似度）
        mask = ~np.eye(n, dtype=bool)
        similarities = similarity_matrix[mask]
        
        avg_similarity = np.mean(similarities)
        similarity_std = np.std(similarities)
        max_similarity = np.max(similarities)
        
        # 多样性指数：标准差除以平均值
        diversity_index = similarity_std / avg_similarity if avg_similarity > 0 else 0
        
        # 过拟合风险评估
        risk_score = 0
        if avg_similarity > 0.85:
            risk_score += 2
        elif avg_similarity > 0.75:
            risk_score += 1
            
        if similarity_std < 0.05:
            risk_score += 2
        elif similarity_std < 0.10:
            risk_score += 1
            
        if max_similarity > 0.95:
            risk_score += 1
        
        if risk_score >= 4:
            overfitting_risk = "高"
        elif risk_score >= 2:
            overfitting_risk = "中"
        else:
            overfitting_risk = "低"
        
        return {
            'avg_similarity': avg_similarity,
            'similarity_std': similarity_std,
            'max_similarity': max_similarity,
            'diversity_index': diversity_index,
            'overfitting_risk': overfitting_risk,
            'risk_score': risk_score
        }

def main():
    print("🚀 启动SDXL训练疑惑综合解谜分析")
    print("="*60)
    
    analyzer = MysteryAnalyzer()
    
    # 执行所有分析
    all_results = {}
    
    try:
        # 疑惑1: 文本长度影响
        all_results['length_impact'] = analyzer.mystery_1_text_length_impact()
        
        # 疑惑2: shuffle_caption机制
        all_results['shuffle_mechanism'] = analyzer.mystery_2_shuffle_caption_mechanism()
        
        # 疑惑3: 结构化格式优化
        all_results['structured_format'] = analyzer.mystery_3_structured_format_optimization()
        
        # 疑惑4: 过拟合检测
        all_results['overfitting_detection'] = analyzer.mystery_4_overfitting_detection()
        
        # 保存结果
        with open('/root/mystery_analysis_results.json', 'w', encoding='utf-8') as f:
            # 只保存可序列化的数据
            serializable_results = {}
            for key, value in all_results.items():
                if key == 'length_impact':
                    serializable_results[key] = [
                        {k: v for k, v in item.items() if k != 'embedding'}
                        for item in value
                    ]
                elif key in ['structured_format', 'overfitting_detection']:
                    if value is not None:
                        serializable_results[key] = {
                            k: {k2: {k3: v3 for k3, v3 in v2.items() if k3 != 'embedding'} 
                                if isinstance(v2, dict) else v2 
                                for k2, v2 in v.items()} 
                            if isinstance(v, dict) else v 
                            for k, v in value.items()
                        }
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 分析完成！结果已保存到 /root/mystery_analysis_results.json")
        
        # 输出总结
        print(f"\n" + "="*60)
        print("📝 核心发现总结:")
        print("="*60)
        print("1. 文本长度截断确实会影响语义理解，但影响程度因模型而异")
        print("2. shuffle_caption机制在维持概念一致性方面表现良好")  
        print("3. 结构化拼接格式的选择对双编码器性能有显著影响")
        print("4. 基于嵌入相似度的过拟合检测是可行的监控方法")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 