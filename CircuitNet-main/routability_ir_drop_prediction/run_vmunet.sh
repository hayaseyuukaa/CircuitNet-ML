#!/bin/bash

# 脚本名称: run_vmunet.sh
# 功能描述: 运行VisionMamba UNet模型进行拥塞预测任务

# 获取当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# 设置工作目录和日志目录
BASE_WORK_DIR="$PROJECT_ROOT/work_dir"
LOG_DIR="$BASE_WORK_DIR/logs"

# 数据路径
DATA_ROOT="$SCRIPT_DIR"
TRAINING_SET_PATH="$DATA_ROOT/training_set"
FILES_PATH="$DATA_ROOT/files"

# 数据集路径
N14_DATA_PATH="$TRAINING_SET_PATH/congestion"
N28_DATA_PATH="$TRAINING_SET_PATH/congestion_trainingset1.0/congestion_trainingset/congestion"
ISPD_DATA_PATH="$TRAINING_SET_PATH/ISPD2015_congestion"

# 确保工作目录和日志目录存在
mkdir -p "$LOG_DIR"

# 时间戳（用于日志文件名）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

cd "$SCRIPT_DIR"
echo "工作目录已切换到: $(pwd)"

# 函数：检查文件路径
check_path() {
    local path=$1
    local description=$2
    if [ ! -d "$path" ]; then
        echo "错误: $description 路径不存在: $path"
        return 1
    else
        echo "$description 路径检查通过: $path"
        return 0
    fi
}

# 函数：检查CSV文件
check_csv() {
    local csv_file=$1
    local description=$2
    if [ ! -f "$csv_file" ]; then
        echo "错误: $description 文件不存在: $csv_file"
        return 1
    else
        local count=$(wc -l < "$csv_file")
        echo "$description 检查通过: $csv_file (包含 $count 条数据)"
        return 0
    fi
}

# 处理命令行参数
if [ $# -eq 0 ]; then
    echo "错误: 未提供参数"
    echo "用法: $0 [train_n14|train_n28|train_ispd|test_n14|test_n28|test_ispd]"
    exit 1
fi

case $1 in
    train_n14)
        WORK_DIR="$BASE_WORK_DIR/congestion_visionmamba"
        mkdir -p "$WORK_DIR"
        LOG_FILE="$LOG_DIR/vmunet_train_n14_${TIMESTAMP}.log"
        
        echo "开始在CircuitNet 2.0 (N14)数据集上训练VisionMamba UNet模型..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$N14_DATA_PATH" "N14数据集" || exit 1
        check_csv "${FILES_PATH}/train_N14.csv" "N14训练数据索引" || exit 1
        
        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "max_iters": 50000,
    "batch_size": 16,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "dataroot": "${N14_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_N14.csv",
    "ann_file_test": "${FILES_PATH}/test_N14.csv",
    "model_type": "VMUNet",
    "in_channels": 3,
    "out_channels": 1,
    "loss_type": "MSELoss",
    "dataset_type": "CongestionDataset",
    "save_path": "${WORK_DIR}",
    "cpu": false,
    "use_pin_memory": false
}
EOF

        # 运行训练脚本并记录日志
        {
            echo "===== CircuitNet 2.0 (N14)数据集训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: VisionMamba UNet"
            echo "工作目录: $WORK_DIR"
            echo "使用ibUNet原项目配置参数"
            echo "====================================="
            echo ""
            
            python "$SCRIPT_DIR/train_simple.py" \
                --arg_file "$PARAMS_FILE"
            
            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee "$LOG_FILE"

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    train_n28)
        WORK_DIR="$BASE_WORK_DIR/congestion_visionmamba_n28"
        mkdir -p "$WORK_DIR"
        LOG_FILE="$LOG_DIR/vmunet_train_n28_${TIMESTAMP}.log"
        
        echo "开始在CircuitNet 1.0 (N28)数据集上训练VisionMamba UNet模型..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$N28_DATA_PATH" "N28数据集" || exit 1
        check_csv "${FILES_PATH}/train_N28.csv" "N28训练数据索引" || exit 1
        
        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "max_iters": 50000,
    "batch_size": 16,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "dataroot": "${N28_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_N28.csv",
    "ann_file_test": "${FILES_PATH}/test_N28.csv",
    "model_type": "VMUNet",
    "in_channels": 3,
    "out_channels": 1,
    "loss_type": "MSELoss",
    "dataset_type": "CongestionDataset",
    "save_path": "${WORK_DIR}",
    "cpu": false,
    "use_pin_memory": false
}
EOF

        # 运行训练脚本并记录日志
        {
            echo "===== CircuitNet 1.0 (N28)数据集训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: VisionMamba UNet"
            echo "工作目录: $WORK_DIR"
            echo "使用ibUNet原项目配置参数"
            echo "====================================="
            echo ""
            
            python "$SCRIPT_DIR/train_simple.py" \
                --arg_file "$PARAMS_FILE"
            
            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee "$LOG_FILE"

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    train_ispd)
        WORK_DIR="$BASE_WORK_DIR/congestion_visionmamba_ispd"
        mkdir -p "$WORK_DIR"
        LOG_FILE="$LOG_DIR/vmunet_train_ispd_${TIMESTAMP}.log"
        
        echo "开始在ISPD2015数据集上训练VisionMamba UNet模型..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$ISPD_DATA_PATH" "ISPD数据集" || exit 1
        check_csv "${FILES_PATH}/train_ISPD2015.csv" "ISPD训练数据索引" || exit 1
        
        # 创建临时参数文件
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
    "model_type": "VMUNet",
    "in_channels": 6,
    "out_channels": 1,
    "loss_type": "MSELoss",
    "dataset_type": "CongestionDataset",
    "early_stopping": true,
    "patience": 10,
    "min_delta": 0.001,
    "dropout_rate": 0.2,
    "save_path": "${WORK_DIR}",
    "cpu": false,
    "use_pin_memory": false
}
EOF

        # 运行训练脚本并记录日志
        {
            echo "===== ISPD2015数据集训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: VisionMamba UNet"
            echo "工作目录: $WORK_DIR"
            echo "使用ibUNet原项目配置参数"
            echo "====================================="
            echo ""
            
            python "$SCRIPT_DIR/train_simple.py" \
                --arg_file "$PARAMS_FILE"
            
            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee "$LOG_FILE"

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    test_*)
        echo "测试功能暂未实现"
        ;;
    *)
        echo "错误: 未知参数 '$1'"
        echo "用法: $0 [train_n14|train_n28|train_ispd|test_n14|test_n28|test_ispd]"
        exit 1
        ;;
esac

echo "操作完成!"
exit 0
