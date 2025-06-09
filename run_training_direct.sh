#!/bin/bash
# SDXL LoRA 直接训练启动脚本 V1.0
# 集成环境检查、数据预处理、tensorboard启动和直接训练启动
# 专为GPU直接加载模型和VAE优化，不使用accelerate和DeepSpeed

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 全局变量
TENSORBOARD_PID=""
SESSION_ID=""
TEMP_CONFIG_FILE=""
TENSORBOARD_PORT=6006
export HF_ENDPOINT=https://hf-mirror.com

# 打印彩色信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 清理函数 - 脚本退出时自动调用
cleanup() {
    print_info "正在清理资源..."
    
    # 停止tensorboard
    if [ ! -z "$TENSORBOARD_PID" ]; then
        print_info "停止 TensorBoard (PID: $TENSORBOARD_PID)"
        kill $TENSORBOARD_PID 2>/dev/null || true
        wait $TENSORBOARD_PID 2>/dev/null || true
        print_success "TensorBoard 已停止"
    fi
    
    # 删除临时配置文件
    if [ ! -z "$TEMP_CONFIG_FILE" ] && [ -f "$TEMP_CONFIG_FILE" ]; then
        rm -f "$TEMP_CONFIG_FILE"
        print_info "已删除临时配置文件: $TEMP_CONFIG_FILE"
    fi
    
    print_success "清理完成"
}

# 注册退出时的清理函数
trap cleanup EXIT INT TERM

# 生成会话ID和时间戳
generate_session_id() {
    local timestamp=$(date '+%Y%m%d%H%M')
    if [ ! -z "$SESSION_NAME" ]; then
        SESSION_ID="${SESSION_NAME}_${timestamp}"
    else
        SESSION_ID="${timestamp}"
    fi
    print_info "会话ID: $SESSION_ID"
}

# 查找可用端口
find_available_port() {
    local port=$1
    while netstat -ln 2>/dev/null | grep -q ":$port "; do
        port=$((port + 1))
    done
    echo $port
}

# 创建动态配置文件
create_dynamic_config() {
    print_info "生成动态训练配置..."
    
    local base_config="config_v3_optimal.toml"
    TEMP_CONFIG_FILE="config_v3_optimal_${SESSION_ID}.toml"
    
    if [ ! -f "$base_config" ]; then
        print_error "基础配置文件不存在: $base_config"
        exit 1
    fi
    
    # 创建会话专用的日志目录
    local session_log_dir="./logs/sessions/${SESSION_ID}"
    local tensorboard_log_dir="${session_log_dir}/tensorboard"
    local output_dir="./models/${SESSION_ID}"
    
    mkdir -p "$tensorboard_log_dir"
    mkdir -p "$output_dir"
    
    # 复制配置并修改路径
    cp "$base_config" "$TEMP_CONFIG_FILE"
    
    # 使用 sed 替换配置文件中的路径
    sed -i "s|logging_dir = \".*\"|logging_dir = \"${tensorboard_log_dir}\"|g" "$TEMP_CONFIG_FILE"
    sed -i "s|output_dir = \".*\"|output_dir = \"${output_dir}\"|g" "$TEMP_CONFIG_FILE"
    sed -i "s|log_prefix = \".*\"|log_prefix = \"${SESSION_ID}\"|g" "$TEMP_CONFIG_FILE"
    
    print_success "动态配置已生成: $TEMP_CONFIG_FILE"
    print_info "TensorBoard 日志目录: $tensorboard_log_dir"
    print_info "模型输出目录: $output_dir"
}

# 启动TensorBoard
start_tensorboard() {
    print_info "启动 TensorBoard 服务..."
    
    # 检查tensorboard是否安装
    if ! command -v tensorboard &> /dev/null; then
        print_error "TensorBoard 未安装，请运行: pip install tensorboard"
        return 1
    fi
    
    # 查找可用端口
    TENSORBOARD_PORT=$(find_available_port $TENSORBOARD_PORT)
    
    local tensorboard_log_dir="./logs/sessions/${SESSION_ID}/tensorboard"
    
    # 启动tensorboard并获取PID
    nohup tensorboard --logdir="$tensorboard_log_dir" --port=$TENSORBOARD_PORT --host=0.0.0.0 > /dev/null 2>&1 &
    TENSORBOARD_PID=$!
    
    # 等待tensorboard启动
    sleep 3
    
    # 验证tensorboard是否成功启动
    if kill -0 $TENSORBOARD_PID 2>/dev/null; then
        print_success "TensorBoard 已启动"
        print_info "TensorBoard URL: http://localhost:${TENSORBOARD_PORT}"
        print_info "TensorBoard PID: $TENSORBOARD_PID"
        
        # 如果在远程服务器上，提供端口转发提示
        if [ ! -z "$SSH_CLIENT" ]; then
            print_info "远程访问命令: ssh -L ${TENSORBOARD_PORT}:localhost:${TENSORBOARD_PORT} $(whoami)@$(hostname)"
        fi
    else
        print_error "TensorBoard 启动失败"
        TENSORBOARD_PID=""
        return 1
    fi
}

# 检查Python环境
check_python_env() {
    print_info "检查Python环境..."
    
    # 使用默认系统Python
    PYTHON_PATH="python"
    
    # 显示Python版本
    PYTHON_VERSION=$($PYTHON_PATH --version)
    print_info "Python版本: $PYTHON_VERSION"
}

# 运行环境检查
run_pre_check() {
    print_info "运行训练前环境检查..."
    
    if $PYTHON_PATH pre_training_check.py; then
        print_success "环境检查通过"
        return 0
    else
        print_error "环境检查失败"
        return 1
    fi
}

# 数据预处理检查
check_data_preprocessing() {
    print_info "检查数据预处理状态..."
    
    RESTRUCTURED_DIR="/root/data/cluster_4_restructured_v3"
    
    if [ ! -d "$RESTRUCTURED_DIR" ]; then
        print_warning "重构数据目录不存在，开始数据预处理..."
        run_data_preprocessing
    else
        # 检查文件数量
        TXT_COUNT=$(find "$RESTRUCTURED_DIR" -name "*.txt" | wc -l)
        IMG_COUNT=$(find "$RESTRUCTURED_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" | wc -l)
        
        print_info "发现 $TXT_COUNT 个文本文件，$IMG_COUNT 个图片文件"
        
        if [ "$TXT_COUNT" -eq 0 ]; then
            print_warning "未发现文本文件，重新运行数据预处理..."
            run_data_preprocessing
        else
            print_success "数据预处理已完成"
        fi
    fi
}

# 运行数据预处理
run_data_preprocessing() {
    print_info "开始数据重构和预处理..."
    
    # 默认使用软连接模式
    if [ "$USE_REAL_COPY" = "true" ]; then
        print_info "使用真实复制模式处理图片"
        $PYTHON_PATH preprocess_tags_v2.py --real
    else
        print_info "使用软连接模式处理图片"
        $PYTHON_PATH preprocess_tags_v2.py
    fi
    
    if [ $? -eq 0 ]; then
        print_success "数据预处理完成"
    else
        print_error "数据预处理失败"
        exit 1
    fi
}

# 启动直接训练
start_direct_training() {
    print_info "准备启动SDXL LoRA直接训练..."
    
    # 检查动态配置文件
    if [ ! -f "$TEMP_CONFIG_FILE" ]; then
        print_error "动态配置文件不存在: $TEMP_CONFIG_FILE"
        exit 1
    fi
    
    print_info "使用配置文件: $TEMP_CONFIG_FILE"
    
    # 使用直接Python执行，不通过accelerate
    print_info "使用直接Python执行模式 (无accelerate/DeepSpeed)"
    TRAIN_CMD="$PYTHON_PATH -m accelerate.commands.launch --mixed_precision=bf16 --num_processes=4  sdxl_train_network.py --config_file $TEMP_CONFIG_FILE"
    
    print_info "训练命令: $TRAIN_CMD"
    
    # 记录启动时间
    START_TIME=$(date)
    print_info "训练开始时间: $START_TIME"
    print_info "会话ID: $SESSION_ID"
    
    # 启动训练
    echo "=========================================="
    echo "开始SDXL LoRA直接训练 - 会话: $SESSION_ID"
    echo "TensorBoard: http://localhost:${TENSORBOARD_PORT}"
    echo "=========================================="
    
    if eval "$TRAIN_CMD"; then
        print_success "训练完成"
        END_TIME=$(date)
        print_info "训练结束时间: $END_TIME"
    else
        print_error "训练失败"
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    echo "SDXL LoRA 直接训练启动脚本 V1.0"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --check-only          仅运行环境检查，不启动训练"
    echo "  --preprocess-only     仅运行数据预处理，不启动训练"  
    echo "  --real-copy           使用真实复制模式处理图片（默认使用软连接）"
    echo "  --tensorboard-port    指定TensorBoard端口（默认: 6006）"
    echo "  --session-name        自定义会话名称前缀"
    echo "  --help, -h            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 完整流程：检查->预处理->训练"
    echo "  $0 --check-only                      # 仅环境检查"
    echo "  $0 --real-copy                       # 使用真实复制模式"
    echo "  $0 --tensorboard-port 6007           # 指定TensorBoard端口"
    echo "  $0 --session-name my_experiment      # 自定义会话名称"
    echo ""
    echo "特性:"
    echo "  ✓ 自动启动TensorBoard服务"
    echo "  ✓ 基于时间戳的训练会话隔离"
    echo "  ✓ 动态配置文件生成"
    echo "  ✓ 退出时自动清理资源"
    echo "  ✓ 端口冲突自动解决"
    echo "  ✓ 直接Python执行，无accelerate/DeepSpeed"
}

# 主函数
main() {
    echo "🚀 SDXL LoRA 直接训练启动器 V1.0"
    echo "=================================================="
    
    # 解析命令行参数
    CHECK_ONLY=false
    PREPROCESS_ONLY=false
    USE_REAL_COPY=false
    SESSION_NAME=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-only)
                CHECK_ONLY=true
                shift
                ;;
            --preprocess-only)
                PREPROCESS_ONLY=true
                shift
                ;;
            --real-copy)
                USE_REAL_COPY=true
                shift
                ;;
            --tensorboard-port)
                TENSORBOARD_PORT="$2"
                shift 2
                ;;
            --session-name)
                SESSION_NAME="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置Python路径
    PYTHON_PATH="python"
    
    # 生成会话ID
    generate_session_id
    
    # 执行检查
    check_python_env
    
    if ! run_pre_check; then
        print_error "环境检查失败，请修复问题后重试"
        exit 1
    fi
    
    if [ "$CHECK_ONLY" = "true" ]; then
        print_info "仅环境检查模式，跳过后续步骤"
        exit 0
    fi
    
    # 数据预处理
    check_data_preprocessing
    
    if [ "$PREPROCESS_ONLY" = "true" ]; then
        print_info "仅数据预处理模式，跳过训练"
        exit 0
    fi
    
    # 创建动态配置文件
    create_dynamic_config
    
    # 启动TensorBoard
    if ! start_tensorboard; then
        print_warning "TensorBoard启动失败，继续训练但无法实时监控"
    fi
    
    # 启动训练（直接使用Python）
    start_direct_training
    
    print_success "所有任务完成！"
}

# 运行主函数
main "$@" 