import os
import json
import argparse
import subprocess
import time
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import sys

class TagProcessingManager:
    """
    标签处理管理器 - 智能批次分片和多进程管理
    """
    
    def __init__(self, source_dir, output_dir, max_workers=None, batch_size=None):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # 智能设置最大工作进程数
        if max_workers is None:
            # 基于GPU数量和CPU核心数智能设置
            cpu_count = multiprocessing.cpu_count()
            # 对于GPU密集型任务，通常4-6个进程比较合适
            self.max_workers = min(6, max(2, cpu_count // 2))
        else:
            self.max_workers = max_workers
            
        # 自动计算最优批次大小
        self.batch_size = self._calculate_optimal_batch_size() if batch_size is None else batch_size
        
        print(f"[MANAGER] 初始化处理管理器")
        print(f"[MANAGER] 源目录: {self.source_dir}")
        print(f"[MANAGER] 输出目录: {self.output_dir}")
        print(f"[MANAGER] 最大并发进程: {self.max_workers}")
        print(f"[MANAGER] 智能批次大小: {self.batch_size}")
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_optimal_batch_size(self):
        """基于数据量和硬件自动计算最优批次大小"""
        # 发现所有txt文件
        txt_files = list(self.source_dir.glob("*.txt"))
        total_files = len(txt_files)
        
        if total_files == 0:
            print("[MANAGER ERROR] 未找到任何txt文件")
            return 50
        
        # 智能批次大小计算逻辑
        if total_files <= 100:
            # 小数据集：每个进程处理10-20个文件
            batch_size = max(10, total_files // self.max_workers)
        elif total_files <= 500:
            # 中等数据集：每个进程处理20-50个文件
            batch_size = max(20, total_files // (self.max_workers * 2))
        elif total_files <= 2000:
            # 大数据集：每个进程处理50-100个文件
            batch_size = max(50, total_files // (self.max_workers * 3))
        else:
            # 超大数据集：每个进程处理100-200个文件
            batch_size = max(100, min(200, total_files // (self.max_workers * 4)))
        
        # 确保批次大小合理
        batch_size = min(batch_size, total_files)
        
        print(f"[MANAGER] 检测到 {total_files} 个文件，计算出最优批次大小: {batch_size}")
        return batch_size

    def _discover_files(self):
        """发现并分析所有需要处理的文件"""
        txt_files = list(self.source_dir.glob("*.txt"))
        
        if not txt_files:
            print("[MANAGER ERROR] 源目录中未找到任何txt文件")
            return []
        
        print(f"[MANAGER] 发现 {len(txt_files)} 个文件待处理")
        
        # 返回文件路径字符串列表
        return [str(f) for f in txt_files]

    def _create_batches(self, file_list):
        """将文件列表分割成批次"""
        batches = []
        
        for i in range(0, len(file_list), self.batch_size):
            batch = file_list[i:i + self.batch_size]
            batches.append(batch)
        
        print(f"[MANAGER] 创建了 {len(batches)} 个批次，每批次约 {self.batch_size} 个文件")
        return batches

    def _run_batch_process(self, batch_files, batch_id):
        """运行单个批次的subprocess"""
        try:
            # 构建命令行参数
            batch_files_str = ",".join(batch_files)
            
            cmd = [
                sys.executable,  # 使用当前Python解释器
                "preprocess_tags_batch.py",
                "--batch-files", batch_files_str,
                "--source-dir", str(self.source_dir),
                "--output-dir", str(self.output_dir),
                "--batch-id", str(batch_id),
                "--max-resolution", "5000"
            ]
            
            print(f"[MANAGER] 启动批次 {batch_id} (文件数: {len(batch_files)})")
            
            # 运行subprocess
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            end_time = time.time()
            process_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"[MANAGER] 批次 {batch_id} 完成 (耗时: {process_time:.1f}s)")
                return {
                    "batch_id": batch_id,
                    "status": "success",
                    "process_time": process_time,
                    "files_count": len(batch_files),
                    "stdout": result.stdout[-500:] if result.stdout else "",  # 只保留最后500字符
                    "stderr": result.stderr[-500:] if result.stderr else ""
                }
            else:
                print(f"[MANAGER ERROR] 批次 {batch_id} 失败 (返回码: {result.returncode})")
                print(f"[MANAGER ERROR] 错误输出: {result.stderr}")
                return {
                    "batch_id": batch_id,
                    "status": "failed",
                    "process_time": process_time,
                    "files_count": len(batch_files),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"[MANAGER ERROR] 批次 {batch_id} 超时")
            return {
                "batch_id": batch_id,
                "status": "timeout",
                "files_count": len(batch_files),
                "error": "Process timeout"
            }
        except Exception as e:
            print(f"[MANAGER ERROR] 批次 {batch_id} 异常: {e}")
            return {
                "batch_id": batch_id,
                "status": "error",
                "files_count": len(batch_files),
                "error": str(e)
            }

    def _collect_batch_results(self):
        """收集所有批次的结果"""
        batch_result_files = list(self.output_dir.glob("batch_*_results.json"))
        
        if not batch_result_files:
            print("[MANAGER ERROR] 未找到任何批次结果文件")
            return None
        
        print(f"[MANAGER] 收集 {len(batch_result_files)} 个批次结果")
        
        all_results = []
        total_stats = {
            "found": 0, "missing": 0, "processed": 0, 
            "errors": 0, "compressed": 0, "total_size_reduction": 0
        }
        
        for result_file in batch_result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                # 累计统计数据
                image_stats = batch_data.get("image_stats", {})
                for key in total_stats:
                    total_stats[key] += image_stats.get(key, 0)
                
                # 收集详细结果
                all_results.extend(batch_data.get("results", []))
                
            except Exception as e:
                print(f"[MANAGER ERROR] 读取批次结果文件 {result_file} 失败: {e}")
        
        return all_results, total_stats

    def _generate_final_report(self, all_results, total_stats, process_info):
        """生成最终的综合报告"""
        if not all_results:
            print("[MANAGER ERROR] 没有有效的处理结果")
            return
        
        # 计算平均值
        import numpy as np
        
        avg_sim1 = np.mean([d['similarity_te1'] for d in all_results if 'similarity_te1' in d])
        avg_sim2 = np.mean([d['similarity_te2'] for d in all_results if 'similarity_te2' in d])
        avg_tokens1 = np.mean([d['tokens_te1'] for d in all_results if 'tokens_te1' in d])
        avg_tokens2 = np.mean([d['tokens_te2'] for d in all_results if 'tokens_te2' in d])
        max_tokens1 = max([d['tokens_te1'] for d in all_results if 'tokens_te1' in d], default=0)
        max_tokens2 = max([d['tokens_te2'] for d in all_results if 'tokens_te2' in d], default=0)
        
        # 计算总体处理性能
        total_process_time = sum([p.get("process_time", 0) for p in process_info])
        successful_batches = len([p for p in process_info if p.get("status") == "success"])
        failed_batches = len([p for p in process_info if p.get("status") != "success"])
        
        print("\n" + "="*80)
        print("🚀 SDXL数据重构 - 多进程并行处理报告")
        print("="*80)
        
        print(f"📊 处理性能统计:")
        print(f"      总文件数: {len(all_results)}")
        print(f"      处理批次数: {len(process_info)}")
        print(f"      成功批次: {successful_batches}")
        print(f"      失败批次: {failed_batches}")
        print(f"      并发进程数: {self.max_workers}")
        print(f"      批次大小: {self.batch_size}")
        print(f"      总处理时间: {total_process_time:.1f} 秒")
        if len(all_results) > 0:
            avg_speed = len(all_results) / total_process_time if total_process_time > 0 else 0
            print(f"      处理速度: {avg_speed:.2f} 文件/秒")
        
        print(f"\n🖼️ 图片处理统计:")
        print(f"      找到图片: {total_stats['found']}")
        print(f"      缺失图片: {total_stats['missing']}")
        print(f"      成功处理: {total_stats['processed']}")
        print(f"      处理错误: {total_stats['errors']}")
        pairing_rate = total_stats['found']/(total_stats['found']+total_stats['missing'])*100 if (total_stats['found']+total_stats['missing']) > 0 else 0
        print(f"      配对成功率: {pairing_rate:.1f}%")
        
        print(f"\n🗜️ 图片压缩统计:")
        print(f"      压缩图片数量: {total_stats['compressed']}")
        compression_rate = total_stats['compressed']/max(total_stats['found'], 1)*100
        print(f"      压缩率: {compression_rate:.1f}%")
        if total_stats['compressed'] > 0:
            avg_size_reduction = total_stats['total_size_reduction'] / total_stats['compressed'] * 100
            print(f"      平均像素减少: {avg_size_reduction:.1f}%")
        
        print(f"\n🔤 语义保真度分析:")
        print(f"      TE1 平均相似度: {avg_sim1:.4f}")
        print(f"      TE2 平均相似度: {avg_sim2:.4f}")
        print(f"      综合平均相似度: {(avg_sim1 + avg_sim2) / 2:.4f}")
        
        print(f"\n📝 Token长度分析:")
        print(f"      TE1 平均Token数: {avg_tokens1:.1f} (最大: {max_tokens1})")
        print(f"      TE2 平均Token数: {avg_tokens2:.1f} (最大: {max_tokens2})")
        
        # 保存最终报告
        final_report = {
            "summary": {
                "processing_mode": "多进程并行处理",
                "total_files": len(all_results),
                "batch_count": len(process_info),
                "successful_batches": successful_batches,
                "failed_batches": failed_batches,
                "max_workers": self.max_workers,
                "batch_size": self.batch_size,
                "total_process_time": total_process_time,
                "processing_speed": len(all_results) / total_process_time if total_process_time > 0 else 0,
                "image_stats": total_stats,
                "avg_similarity_te1": float(avg_sim1),
                "avg_similarity_te2": float(avg_sim2),
                "avg_tokens_te1": float(avg_tokens1),
                "max_tokens_te1": int(max_tokens1),
                "avg_tokens_te2": float(avg_tokens2),
                "max_tokens_te2": int(max_tokens2)
            },
            "batch_info": process_info,
            "detailed_results": all_results
        }
        
        final_report_path = self.output_dir / "final_processing_report.json"
        with open(final_report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] 多进程处理完成！最终报告已保存至: {final_report_path}")
        print("="*80)

    def run(self):
        """执行完整的多进程处理流程"""
        print(f"[MANAGER] 开始多进程标签处理...")
        
        # 1. 发现文件
        file_list = self._discover_files()
        if not file_list:
            return
        
        # 2. 创建批次
        batches = self._create_batches(file_list)
        
        # 3. 并行处理批次
        process_info = []
        
        print(f"[MANAGER] 开始并行处理 {len(batches)} 个批次...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(self._run_batch_process, batch, i): i 
                for i, batch in enumerate(batches)
            }
            
            # 使用tqdm显示总体进度
            with tqdm(total=len(batches), desc="批次处理进度", unit="批次") as pbar:
                for future in as_completed(future_to_batch):
                    batch_id = future_to_batch[future]
                    try:
                        result = future.result()
                        process_info.append(result)
                        
                        # 更新进度条描述
                        if result["status"] == "success":
                            pbar.set_postfix({
                                "成功": len([p for p in process_info if p.get("status") == "success"]),
                                "失败": len([p for p in process_info if p.get("status") != "success"])
                            })
                        
                    except Exception as e:
                        print(f"[MANAGER ERROR] 批次 {batch_id} 执行异常: {e}")
                        process_info.append({
                            "batch_id": batch_id,
                            "status": "exception",
                            "error": str(e)
                        })
                    
                    pbar.update(1)
        
        # 4. 收集结果
        print(f"[MANAGER] 收集处理结果...")
        results_data = self._collect_batch_results()
        if results_data is None:
            print("[MANAGER ERROR] 未能收集到有效结果")
            return
        
        all_results, total_stats = results_data
        
        # 5. 生成最终报告
        self._generate_final_report(all_results, total_stats, process_info)
        
        # 6. 清理临时批次结果文件
        print(f"[MANAGER] 清理临时文件...")
        for result_file in self.output_dir.glob("batch_*_results.json"):
            try:
                result_file.unlink()
            except Exception as e:
                print(f"[MANAGER WARNING] 清理文件 {result_file} 失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多进程标签处理管理器")
    parser.add_argument("--source-dir", required=True,
                       help="源数据目录")
    parser.add_argument("--output-dir", required=True,
                       help="输出目录")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="最大并发进程数 (默认: 自动检测)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="批次大小 (默认: 自动计算)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    print(f"[MANAGER] 启动多进程标签处理管理器")
    print(f"[MANAGER] 源目录: {args.source_dir}")
    print(f"[MANAGER] 输出目录: {args.output_dir}")
    
    # 创建管理器并执行
    manager = TagProcessingManager(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    start_time = time.time()
    manager.run()
    end_time = time.time()
    
    print(f"\n[MANAGER] 总执行时间: {end_time - start_time:.1f} 秒")
    print(f"[MANAGER] 多进程处理完成！") 