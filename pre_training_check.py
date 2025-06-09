#!/usr/bin/env python3
"""
SDXL LoRA 训练前环境检查脚本 V2.1
检查训练环境、依赖项、数据集、模型文件和TensorBoard服务可用性
注意：已移除DeepSpeed支持
"""

import os
import sys
import json
import psutil
import subprocess
import socket
from pathlib import Path
from datetime import datetime
import importlib.util

def print_section(title):
    """打印格式化的章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_python_environment():
    """检查Python环境和必要包"""
    print_section("🐍 Python环境检查")
    
    print(f"✓ Python版本: {sys.version}")
    
    # 训练所需的关键包列表
    packages = [
        'torch', 'torchvision', 'transformers', 'accelerate', 'diffusers',
        'tensorboard', 'numpy', 'PIL', 'tqdm', 'sklearn', 'psutil', 'toml',
        'cv2', 'safetensors'  # 移除bitsandbytes的强制检查
    ]
    
    missing_packages = []
    for package_name in packages:
        try:
            if package_name == 'cv2':
                import cv2
                print(f"✓ opencv-python: {cv2.__version__}")
            elif package_name == 'PIL':
                from PIL import Image
                print(f"✓ PIL: {Image.__version__}")
            elif package_name == 'sklearn':
                import sklearn
                print(f"✓ sklearn: {sklearn.__version__}")
            else:
                module = importlib.import_module(package_name)
                if hasattr(module, '__version__'):
                    print(f"✓ {package_name}: {module.__version__}")
                else:
                    print(f"✓ {package_name}: (已安装)")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name}: 未安装")
        except Exception as e:
            # 捕获cv2等包可能的其他错误
            print(f"⚠️  {package_name}: 可用但有警告 - {str(e)[:50]}...")
    
    # 单独检查bitsandbytes（非强制，避免CUDA编译错误）
    try:
        # 使用spec方式检查，避免实际导入
        spec = importlib.util.find_spec('bitsandbytes')
        if spec is not None:
            print("✓ bitsandbytes: 已安装 (可选优化包)")
        else:
            print("⚠️  bitsandbytes: 未安装 (可选优化包，不影响基本训练)")
    except Exception as e:
        print("⚠️  bitsandbytes: 检查时出现问题 (可选优化包，不影响基本训练)")
    
    if missing_packages:
        print(f"\n⚠️  缺失包: {', '.join(missing_packages)}")
        return False
    return True

def check_torch_cuda():
    """检查PyTorch和CUDA配置"""
    print_section("🔥 PyTorch & CUDA检查")
    
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA版本: {torch.version.cuda}")
            print(f"✓ cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_memory_used = torch.cuda.memory_allocated(i) / 1024**3
                gpu_memory_free = gpu_memory - gpu_memory_used
                
                print(f"✓ GPU {i}: {gpu_name}")
                print(f"  - 显存总量: {gpu_memory:.1f} GB")
                print(f"  - 已使用: {gpu_memory_used:.1f} GB")
                print(f"  - 可用: {gpu_memory_free:.1f} GB")
                
                if gpu_memory_free < 8.0:
                    print(f"  ⚠️  可用显存不足，建议至少8GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练（非常慢）")
            return False
        
        return True
        
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    except Exception as e:
        print(f"✗ PyTorch检查失败: {e}")
        return False

def check_tensorboard():
    """检查TensorBoard可用性"""
    print_section("📊 TensorBoard检查")
    
    try:
        # 检查tensorboard命令是否可用
        result = subprocess.run(['tensorboard', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ TensorBoard版本: {version}")
        else:
            print("✗ TensorBoard命令不可用")
            return False
    except FileNotFoundError:
        print("✗ TensorBoard未安装或不在PATH中")
        return False
    except subprocess.TimeoutExpired:
        print("✗ TensorBoard检查超时")
        return False
    except Exception as e:
        print(f"✗ TensorBoard检查失败: {e}")
        return False
    
    # 检查端口可用性
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    default_ports = [6006, 6007, 6008, 6009, 6010]
    available_ports = [port for port in default_ports if is_port_available(port)]
    
    print(f"✓ 可用端口: {available_ports}")
    if not available_ports:
        print("⚠️  常用TensorBoard端口都被占用，脚本将自动寻找可用端口")
    
    return True

def check_model_files():
    """检查必要的模型文件"""
    print_section("🤖 模型文件检查")
    
    model_files = {
        "SDXL基础模型": "/root/checkpoints/sd_xl_base_1.0.safetensors",
        "SDXL VAE": "/root/checkpoints/sdxl_vae.safetensors",
        "TE1模型": "/root/text_encoder/clip-vit-large-patch14",
        "TE2模型": "/root/text_encoder/CLIP-ViT-B-32-laion2B-s34B-b79K"
    }
    
    all_exist = True
    for name, path in model_files.items():
        if os.path.exists(path):
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024**3)
                print(f"✓ {name}: {path} ({size:.1f} GB)")
            else:
                print(f"✓ {name}: {path} (目录)")
        else:
            print(f"✗ {name}: {path} - 不存在")
            all_exist = False
    
    return all_exist

def check_dataset():
    """检查数据集"""
    print_section("📁 数据集检查")
    
    # 检查原始数据集
    original_data_dir = "/root/data/cluster_4"
    restructured_data_dir = "/root/data/cluster_4_restructured_v3"
    
    if not os.path.exists(original_data_dir):
        print(f"✗ 原始数据目录不存在: {original_data_dir}")
        return False
    
    # 统计原始数据
    original_txt_files = list(Path(original_data_dir).glob("*.txt"))
    original_img_files = list(Path(original_data_dir).glob("*.jpg")) + \
                        list(Path(original_data_dir).glob("*.jpeg")) + \
                        list(Path(original_data_dir).glob("*.png")) + \
                        list(Path(original_data_dir).glob("*.webp"))
    
    print(f"✓ 原始数据目录: {original_data_dir}")
    print(f"  - 文本文件: {len(original_txt_files)}")
    print(f"  - 图片文件: {len(original_img_files)}")
    
    # 检查重构数据
    if os.path.exists(restructured_data_dir):
        restructured_txt_files = list(Path(restructured_data_dir).glob("*.txt"))
        restructured_img_files = list(Path(restructured_data_dir).glob("*.jpg")) + \
                                list(Path(restructured_data_dir).glob("*.jpeg")) + \
                                list(Path(restructured_data_dir).glob("*.png")) + \
                                list(Path(restructured_data_dir).glob("*.webp"))
        
        print(f"✓ 重构数据目录: {restructured_data_dir}")
        print(f"  - 文本文件: {len(restructured_txt_files)}")
        print(f"  - 图片文件: {len(restructured_img_files)}")
        
        # 计算配对率
        if len(original_txt_files) > 0:
            pairing_rate = len(restructured_txt_files) / len(original_txt_files) * 100
            print(f"  - 数据配对率: {pairing_rate:.1f}%")
            
            if pairing_rate < 90:
                print("  ⚠️  配对率较低，建议重新运行数据预处理")
        
        return len(restructured_txt_files) > 0
    else:
        print(f"⚠️  重构数据目录不存在: {restructured_data_dir}")
        print("   需要运行数据预处理")
        return False

def check_training_config():
    """检查训练配置文件"""
    print_section("⚙️  训练配置检查")
    
    config_files = [
        "config_v3_optimal.toml",
        "prompts.txt"
    ]
    
    all_exist = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ 配置文件存在: {config_file}")
        else:
            print(f"✗ 配置文件不存在: {config_file}")
            all_exist = False
    
    return all_exist

def check_system_resources():
    """检查系统资源"""
    print_section("💻 系统资源检查")
    
    # CPU信息
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"✓ CPU核心数: {cpu_count}")
    print(f"✓ CPU使用率: {cpu_percent}%")
    
    # 内存信息
    memory = psutil.virtual_memory()
    memory_total = memory.total / (1024**3)
    memory_available = memory.available / (1024**3)
    memory_percent = memory.percent
    
    print(f"✓ 内存总量: {memory_total:.1f} GB")
    print(f"✓ 可用内存: {memory_available:.1f} GB")
    print(f"✓ 内存使用率: {memory_percent}%")
    
    if memory_available < 16:
        print("⚠️  可用内存不足，建议至少16GB")
    
    # 磁盘空间
    disk = psutil.disk_usage('/')
    disk_total = disk.total / (1024**3)
    disk_free = disk.free / (1024**3)
    disk_percent = (disk.used / disk.total) * 100
    
    print(f"✓ 磁盘总量: {disk_total:.1f} GB")
    print(f"✓ 可用空间: {disk_free:.1f} GB")
    print(f"✓ 磁盘使用率: {disk_percent:.1f}%")
    
    if disk_free < 50:
        print("⚠️  磁盘空间不足，建议至少50GB可用空间")
    
    return True

def generate_training_command():
    """生成推荐的训练命令"""
    print_section("🚀 推荐训练命令")
    
    base_cmd = "./start_training.sh"
    
    print("基础命令:")
    print(f"  {base_cmd}")
    print()
    
    print("常用选项:")
    print(f"  {base_cmd} --check-only                    # 仅环境检查")
    print(f"  {base_cmd} --preprocess-only              # 仅数据预处理")
    print(f"  {base_cmd} --real-copy                    # 使用真实复制")
    print(f"  {base_cmd} --session-name my_exp         # 自定义会话名")
    print(f"  {base_cmd} --tensorboard-port 6007       # 指定端口")

def check_deepspeed():
    """检查DeepSpeed配置 - 已禁用"""
    print_section("⚡ DeepSpeed检查")
    
    print("❌ DeepSpeed支持已被移除（由于兼容性问题）")
    print("✓ 将使用标准PyTorch训练模式")
    
    return True  # 返回True，因为我们不再依赖DeepSpeed

def main():
    """主检查函数"""
    print("🔍 SDXL LoRA 训练环境检查")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    checks = []
    
    # 执行各项检查 - 移除DeepSpeed检查
    check_results = {
        "Python环境": check_python_environment(),
        "PyTorch&CUDA": check_torch_cuda(),
        "TensorBoard": check_tensorboard(),
        "模型文件": check_model_files(),
        "数据集": check_dataset(),
        "训练配置": check_training_config(),
        "系统资源": check_system_resources()
    }
    
    # 统计结果
    passed = sum(check_results.values())
    total = len(check_results)
    
    # 显示摘要
    print_section("📋 检查摘要")
    for check_name, result in check_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {check_name:<15} {status}")
    
    print(f"\n总计: {passed}/{total} 项检查通过")
    print("📌 注意：DeepSpeed支持已移除，使用标准PyTorch训练")
    
    if passed == total:
        print("\n🎉 所有检查通过！可以开始训练。")
        generate_training_command()
        return 0
    elif passed >= total - 2:
        print("\n⚠️  大部分检查通过，可以尝试训练，但可能遇到问题。")
        generate_training_command()
        return 0
    else:
        print("\n❌ 多项检查失败，请修复问题后再开始训练。")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n用户中断检查")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n检查过程中发生错误: {e}")
        sys.exit(1) 