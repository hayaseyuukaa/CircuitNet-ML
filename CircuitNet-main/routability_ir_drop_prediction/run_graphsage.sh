#!/bin/bash

# 脚本名称: run_graphsage.sh
# 功能描述: 运行GraphSAGE模型进行拥塞预测任务
# 用法示例: 
#   ./run_graphsage.sh train_n14    # 在CircuitNet 2.0 (N14)数据集上训练
#   ./run_graphsage.sh train_n28    # 在CircuitNet 1.0 (N28)数据集上训练
#   ./run_graphsage.sh train_ispd   # 在ISPD2015数据集上训练
#   ./run_graphsage.sh test_n14     # 在CircuitNet 2.0 (N14)数据集上测试
#   ./run_graphsage.sh test_n28     # 在CircuitNet 1.0 (N28)数据集上测试
#   ./run_graphsage.sh test_ispd    # 在ISPD2015数据集上测试

# 强制切换到脚本所在目录，确保路径正确
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 获取项目根目录（假设是上一级目录）
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "工作目录已切换到: $(pwd)"

# 设置工作目录和日志目录
BASE_WORK_DIR="$PROJECT_ROOT/work_dir"
LOG_DIR="$BASE_WORK_DIR/logs"

# 数据路径
DATA_ROOT="$SCRIPT_DIR"
TRAINING_SET_PATH="$DATA_ROOT/training_set"
FILES_PATH="$DATA_ROOT/files"

# 显示执行环境信息
echo "调试: 执行环境信息"
echo "脚本路径: $SCRIPT_DIR" 
echo "当前工作目录: $(pwd)"
echo "当前用户: $(whoami)"
echo "Python版本: $(python --version 2>&1 || echo '未安装Python')"

# 数据集路径
N14_DATA_PATH="$TRAINING_SET_PATH/congestion"
N28_DATA_PATH="$TRAINING_SET_PATH/congestion_trainingset1.0/congestion_trainingset/congestion"
ISPD_DATA_PATH="$TRAINING_SET_PATH/ISPD2015_congestion"

# 确保工作目录和日志目录存在
mkdir -p "$LOG_DIR"

# 时间戳（用于日志文件名）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 处理命令行参数
if [ $# -eq 0 ]; then
    echo "错误: 未提供参数"
    echo "用法: $0 [train_n14|train_n28|train_ispd|test_n14|test_n28|test_ispd]"
    exit 1
fi

# 函数：检查文件路径
check_path() {
    local path=$1
    local description=$2
    
    if [ ! -d "$path" ]; then
        echo "错误: $description 路径不存在: $path"
        return 1
    fi
    
    echo "$description 路径检查通过: $path"
    return 0
}

# 函数：检查CSV文件
check_csv() {
    local csv_file=$1
    local description=$2
    
    if [ ! -f "$csv_file" ]; then
        echo "错误: $description 不存在: $csv_file"
        return 1
    fi
    
    local count=$(wc -l < "$csv_file")
    echo "$description 检查通过: $csv_file (包含 $count 条数据)"
    return 0
}

# 激活conda环境
echo "调试: 准备激活conda环境..."

# 确保conda命令可用
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "错误: 找不到conda，请确保已安装Anaconda或Miniconda"
        exit 1
    fi
fi

. "$HOME/anaconda3/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null
conda activate circuitnet

# 验证环境激活成功
if [ "$(conda info --envs | grep '*' | awk '{print $1}')" != "circuitnet" ]; then
    echo "错误: 无法激活circuitnet环境"
    exit 1
else
    echo "成功激活 circuitnet 环境"
    python --version
fi

# 根据参数执行相应操作
case "$1" in
    train_n14)
        WORK_DIR="$BASE_WORK_DIR/congestion_graphsage"
        mkdir -p "$WORK_DIR"
        LOG_FILE="$LOG_DIR/graphsage_train_n14_${TIMESTAMP}.log"
        
        echo "开始在CircuitNet 2.0 (N14)数据集上训练GraphSAGE模型..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$N14_DATA_PATH" "N14数据集" || exit 1
        check_csv "${FILES_PATH}/train_N14.csv" "N14训练数据索引" || exit 1
        
        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "max_iters": 50000,
    "batch_size": 4,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "model_type": "GraphSAGECongestion",
    "in_channels": 3,
    "out_channels": 1,
    "hidden_channels": 64,
    "num_layers": 2,
    "dropout": 0.6,
    "aggr": "mean",
    "dataroot": "${N14_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_N14.csv",
    "ann_file_test": "${FILES_PATH}/test_N14.csv",
    "save_path": "${WORK_DIR}",
    "cpu": false
}
EOF
        
        # 运行训练脚本并记录日志
        {
            echo "===== CircuitNet 2.0 (N14)数据集GraphSAGE训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: GraphSAGECongestion"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $N14_DATA_PATH"
            echo "====================================="
            
            python "$SCRIPT_DIR/train_simple.py" \
                --arg_file "$PARAMS_FILE" 2>&1
            
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee "$LOG_FILE"
        
        rm "$PARAMS_FILE"
        ;;
    train_n28)
        WORK_DIR="$BASE_WORK_DIR/congestion_graphsage_n28"
        mkdir -p "$WORK_DIR"
        LOG_FILE="$LOG_DIR/graphsage_train_n28_${TIMESTAMP}.log"
        
        echo "开始在CircuitNet 1.0 (N28)数据集上训练GraphSAGE模型..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$N28_DATA_PATH" "N28数据集" || exit 1
        check_csv "${FILES_PATH}/train_N28.csv" "N28训练数据索引" || exit 1
        
        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "max_iters": 50000,
    "batch_size": 4,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "model_type": "GraphSAGECongestion",
    "in_channels": 3,
    "out_channels": 1,
    "hidden_channels": 64,
    "num_layers": 2,
    "dropout": 0.6,
    "aggr": "mean",
    "dataroot": "${N28_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_N28.csv",
    "ann_file_test": "${FILES_PATH}/test_N28.csv",
    "save_path": "${WORK_DIR}",
    "cpu": false
}
EOF
        
        # 运行训练脚本并记录日志
        {
            echo "===== CircuitNet 1.0 (N28)数据集GraphSAGE训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: GraphSAGECongestion"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $N28_DATA_PATH"
            echo "====================================="
            
            python "$SCRIPT_DIR/train_simple.py" \
                --arg_file "$PARAMS_FILE" 2>&1
            
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee "$LOG_FILE"
        
        rm "$PARAMS_FILE"
        ;;
    train_ispd)
        WORK_DIR="$BASE_WORK_DIR/congestion_graphsage_ispd"
        mkdir -p "$WORK_DIR"
        LOG_FILE="$LOG_DIR/graphsage_train_ispd_${TIMESTAMP}.log"
        
        echo "开始在ISPD2015数据集上训练GraphSAGE模型..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$ISPD_DATA_PATH" "ISPD数据集" || exit 1
        check_csv "${FILES_PATH}/train_ISPD2015.csv" "ISPD训练数据索引" || exit 1
        
        # 创建临时参数文件 - 使用ibUNet原项目配置
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "max_iters": 20000,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "dataroot": "${ISPD_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_ISPD2015.csv",
    "ann_file_test": "${FILES_PATH}/test_ISPD2015.csv",
    "model_type": "GraphSAGECongestion",
    "in_channels": 6,
    "out_channels": 1,
    "hidden_channels": 64,
    "num_layers": 2,
    "dropout": 0.6,
    "aggr": "mean",
    "loss_type": "MSELoss",
    "aug_pipeline": ["Flip"],
    "dataset_type": "CongestionDataset",
    "early_stopping": true,
    "patience": 10,
    "min_delta": 0.001,
    "dropout_rate": 0.2,
    "save_path": "${WORK_DIR}",
    "cpu": false,
    "use_cosine_lr": true,
    "min_lr_ratio": 0.01,
    "use_pin_memory": false
}
EOF

        # 运行统一的训练脚本并记录日志
        {
            echo "===== ISPD2015数据集GraphSAGE训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: GraphSAGECongestion"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $ISPD_DATA_PATH"
            echo "使用ibUNet原项目配置参数"
            echo "====================================="

            python "$SCRIPT_DIR/train_simple.py" \
                --arg_file "$PARAMS_FILE"

            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee "$LOG_FILE"

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    test_n14)
        WORK_DIR="$BASE_WORK_DIR/congestion_graphsage"
        LOG_FILE="$LOG_DIR/graphsage_test_n14_${TIMESTAMP}.log"
        
        echo "测试GraphSAGE模型（在CircuitNet 2.0 (N14)数据集上）..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$N14_DATA_PATH" "N14数据集" || exit 1
        check_csv "${FILES_PATH}/test_N14.csv" "N14测试数据索引" || exit 1
        
        # 使用最新的检查点文件
        LATEST_CHECKPOINT=$(find "$WORK_DIR" -name "*.pth" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
        
        if [ -z "$LATEST_CHECKPOINT" ]; then
            echo "错误: 未找到检查点文件，请先训练模型"
            exit 1
        fi
        
        echo "使用检查点: $LATEST_CHECKPOINT"
        
        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "test_mode": true,
    "pretrained": "$LATEST_CHECKPOINT",
    "model_type": "GraphSAGECongestion",
    "in_channels": 3,
    "out_channels": 1,
    "hidden_channels": 64,
    "num_layers": 2,
    "dropout": 0.6,
    "aggr": "mean",
    "load_state_dict": true,
    "dataroot": "${N14_DATA_PATH}",
    "ann_file_test": "${FILES_PATH}/test_N14.csv",
    "save_path": "${WORK_DIR}",
    "cpu": false,
    "eval_metric": ["NRMS", "SSIM", "PSNR", "peak_nrms"]
}
EOF
        
        # 运行测试脚本并记录日志
        {
            echo "===== CircuitNet 2.0 (N14)数据集GraphSAGE测试日志 ====="
            echo "开始时间: $(date)"
            echo "模型: GraphSAGECongestion"
            echo "检查点: $LATEST_CHECKPOINT"
            echo "数据集路径: $N14_DATA_PATH"
            echo "====================================="
            
            python "$SCRIPT_DIR/test_simple.py" \
                --arg_file "$PARAMS_FILE" 2>&1
            
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee "$LOG_FILE"
        
        rm "$PARAMS_FILE"
        ;;
    test_n28)
        WORK_DIR="$BASE_WORK_DIR/congestion_graphsage_n28"
        LOG_FILE="$LOG_DIR/graphsage_test_n28_${TIMESTAMP}.log"
        
        echo "测试GraphSAGE模型（在CircuitNet 1.0 (N28)数据集上）..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$N28_DATA_PATH" "N28数据集" || exit 1
        check_csv "${FILES_PATH}/test_N28.csv" "N28测试数据索引" || exit 1
        
        # 使用最新的检查点文件
        LATEST_CHECKPOINT=$(find "$WORK_DIR" -name "*.pth" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
        
        if [ -z "$LATEST_CHECKPOINT" ]; then
            echo "错误: 未找到检查点文件，请先训练模型"
            exit 1
        fi
        
        echo "使用检查点: $LATEST_CHECKPOINT"
        
        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "test_mode": true,
    "pretrained": "$LATEST_CHECKPOINT",
    "model_type": "GraphSAGECongestion",
    "in_channels": 3,
    "out_channels": 1,
    "hidden_channels": 64,
    "num_layers": 2,
    "dropout": 0.6,
    "aggr": "mean",
    "load_state_dict": true,
    "dataroot": "${N28_DATA_PATH}",
    "ann_file_test": "${FILES_PATH}/test_N28.csv",
    "save_path": "${WORK_DIR}",
    "cpu": false,
    "eval_metric": ["NRMS", "SSIM", "PSNR", "peak_nrms"]
}
EOF
        
        # 运行测试脚本并记录日志
        {
            echo "===== CircuitNet 1.0 (N28)数据集GraphSAGE测试日志 ====="
            echo "开始时间: $(date)"
            echo "模型: GraphSAGECongestion"
            echo "检查点: $LATEST_CHECKPOINT"
            echo "数据集路径: $N28_DATA_PATH"
            echo "====================================="
            
            python "$SCRIPT_DIR/test_simple.py" \
                --arg_file "$PARAMS_FILE" 2>&1
            
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee "$LOG_FILE"
        
        rm "$PARAMS_FILE"
        ;;
    test_ispd)
        WORK_DIR="$BASE_WORK_DIR/congestion_graphsage_ispd"
        LOG_FILE="$LOG_DIR/graphsage_test_ispd_${TIMESTAMP}.log"
        
        echo "测试GraphSAGE模型（在ISPD2015数据集上）..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$ISPD_DATA_PATH" "ISPD数据集" || exit 1
        check_csv "${FILES_PATH}/test_ISPD2015.csv" "ISPD测试数据索引" || exit 1
        
        # 使用最新的检查点文件
        LATEST_CHECKPOINT=$(find "$WORK_DIR" -name "*.pth" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
        
        if [ -z "$LATEST_CHECKPOINT" ]; then
            echo "错误: 未找到检查点文件，请先训练模型"
            exit 1
        fi
        
        echo "使用检查点: $LATEST_CHECKPOINT"
        
        # 运行测试脚本并记录日志
        {
            echo "===== ISPD2015数据集GraphSAGE测试日志 ====="
            echo "开始时间: $(date)"
            echo "模型: GraphSAGECongestion"
            echo "检查点: $LATEST_CHECKPOINT"
            echo "数据集路径: $ISPD_DATA_PATH"
            echo "====================================="
            
            # 创建临时参数文件用于test_simple.py
            PARAMS_FILE=$(mktemp)
            cat > "$PARAMS_FILE" << EOF
{
    "test_mode": true,
    "pretrained": "$LATEST_CHECKPOINT",
    "model_type": "GraphSAGECongestion",
    "in_channels": 6,
    "out_channels": 1,
    "load_state_dict": true,
    "dataroot": "${ISPD_DATA_PATH}",
    "ann_file_test": "${FILES_PATH}/test_ISPD2015.csv",
    "save_path": "${WORK_DIR}/results_ispd",
    "cpu": false,
    "eval_metric": ["NRMS", "SSIM", "PSNR", "peak_nrms"]
}
EOF

            python "$SCRIPT_DIR/test_simple.py" \
                --arg_file "$PARAMS_FILE" 2>&1

            # 删除临时参数文件
            rm "$PARAMS_FILE"
            
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee "$LOG_FILE"
        
        rm "$PARAMS_FILE"
        ;;
    *)
        echo "错误: 无效参数 '$1'"
        echo "用法: $0 [train_n14|train_n28|train_ispd|test_n14|test_n28|test_ispd]"
        exit 1
        ;;
esac

echo "GraphSAGE脚本执行完成！" 