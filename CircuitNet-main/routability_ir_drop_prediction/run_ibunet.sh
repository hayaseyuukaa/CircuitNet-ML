#!/bin/bash

# 脚本名称: run_ibunet.sh
# 功能描述: 运行ibUNet模型进行拥塞预测任务
# 模型说明: ibUNet (Inception Boosted U-Net) - 轻量级Inception增强U-Net神经网络
# 用法示例: 
#   ./run_ibunet.sh train_n14    # 在CircuitNet 2.0 (N14)数据集上训练
#   ./run_ibunet.sh train_n28    # 在CircuitNet 1.0 (N28)数据集上训练
#   ./run_ibunet.sh train_ispd   # 在ISPD2015数据集上训练
#   ./run_ibunet.sh test_n14     # 在CircuitNet 2.0 (N14)数据集上测试
#   ./run_ibunet.sh test_n28     # 在CircuitNet 1.0 (N28)数据集上测试
#   ./run_ibunet.sh test_ispd    # 在ISPD2015数据集上测试

# 获取当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 获取项目根目录（假设是上一级目录）
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

cd "$SCRIPT_DIR"
echo "工作目录已切换到: $(pwd)"

# 设置工作目录和日志目录
BASE_WORK_DIR="$PROJECT_ROOT/work_dir"
LOG_DIR="${BASE_WORK_DIR}/logs"

# 数据路径
DATA_ROOT="${SCRIPT_DIR}"
TRAINING_SET_PATH="${DATA_ROOT}/training_set"
FILES_PATH="${DATA_ROOT}/files"

# 数据集路径
N14_DATA_PATH="${TRAINING_SET_PATH}/congestion"
N28_DATA_PATH="${TRAINING_SET_PATH}/congestion_trainingset1.0/congestion_trainingset/congestion"
ISPD_DATA_PATH="${TRAINING_SET_PATH}/ISPD2015_congestion"

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
source ~/anaconda3/etc/profile.d/conda.sh
conda activate circuitnet
if [ $? -ne 0 ]; then
    echo "错误: 无法激活circuitnet环境"
    echo "请确保已创建并配置环境: conda env create -f circuitnet_env.yml"
    exit 1
fi

# 根据参数执行相应操作
case "$1" in
    train_n14)
        WORK_DIR="${BASE_WORK_DIR}/congestion_ibunet"
        mkdir -p "$WORK_DIR"
        LOG_FILE="${LOG_DIR}/ibunet_train_n14_${TIMESTAMP}.log"
        
        echo "开始在CircuitNet 2.0 (N14)数据集上训练ibUNet模型..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$N14_DATA_PATH" "N14数据集" || exit 1
        check_csv "${FILES_PATH}/train_N14.csv" "N14训练数据索引" || exit 1
        echo "N14特征目录: $(ls -l ${N14_DATA_PATH}/feature | wc -l) 个文件"
        echo "N14标签目录: $(ls -l ${N14_DATA_PATH}/label | wc -l) 个文件"
        
        # 定义训练参数 - 与ibUNet原项目完全一致的配置
        BATCH_SIZE=16
        LR="2e-4"
        MAX_ITERS=200000
        WEIGHT_DECAY="1e-4"

        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "max_iters": $MAX_ITERS,
    "batch_size": $BATCH_SIZE,
    "lr": $LR,
    "weight_decay": $WEIGHT_DECAY,
    "dataroot": "${N14_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_N14.csv",
    "ann_file_test": "${FILES_PATH}/test_N14.csv",
    "model_type": "Congestion_Prediction_Net",
    "in_channels": 3,
    "out_channels": 1,
    "loss_type": "MSELoss",
    "aug_pipeline": ["Flip"],
    "dataset_type": "CongestionDataset",
    "save_path": "${WORK_DIR}",
    "use_cosine_lr": true,
    "warmup_iters": 1000,
    "min_lr_ratio": 0.01,
    "use_pin_memory": false
}
EOF
        
        # 输出训练信息到终端和日志文件
        {
            echo "===== CircuitNet 2.0 (N14)数据集训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: ibUNet (Inception Boosted U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $N14_DATA_PATH"
            echo "训练数据文件: ${FILES_PATH}/train_N14.csv"
            echo "使用ibUNet原项目配置参数"
            echo "====================================="
            echo ""
            
            # 运行简洁的训练脚本
            python "$SCRIPT_DIR/train_simple.py" \
                --arg_file "$PARAMS_FILE"
                
            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } > >(tee "$LOG_FILE") 2>&1

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    train_n28)
        WORK_DIR="${BASE_WORK_DIR}/congestion_ibunet_n28"
        mkdir -p "$WORK_DIR"
        LOG_FILE="${LOG_DIR}/ibunet_train_n28_${TIMESTAMP}.log"
        
        # 检查外部传入的LOG_FILE变量 - 如果由run_model.sh传入了LOG_FILE，则使用该文件
        if [ -n "$EXTERNAL_LOG_FILE" ]; then
            LOG_FILE="$EXTERNAL_LOG_FILE"
            echo "使用外部指定的日志文件: $LOG_FILE"
        fi
        
        echo "开始在CircuitNet 1.0 (N28)数据集上训练ibUNet模型..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$N28_DATA_PATH" "N28数据集" || exit 1
        check_path "${N28_DATA_PATH}/feature" "N28特征目录" || exit 1
        check_path "${N28_DATA_PATH}/label" "N28标签目录" || exit 1
        check_csv "${FILES_PATH}/train_N28.csv" "N28训练数据索引" || exit 1
        echo "N28特征目录: $(ls -l ${N28_DATA_PATH}/feature | wc -l) 个文件"
        echo "N28标签目录: $(ls -l ${N28_DATA_PATH}/label | wc -l) 个文件"
        
        # 创建临时参数文件 - 与ibUNet原项目完全一致的配置
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "max_iters": 200000,
    "batch_size": 16,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "dataroot": "${N28_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_N28.csv",
    "ann_file_test": "${FILES_PATH}/test_N28.csv",
    "model_type": "Congestion_Prediction_Net",
    "in_channels": 3,
    "out_channels": 1,
    "loss_type": "MSELoss",
    "aug_pipeline": ["Flip"],
    "dataset_type": "CongestionDataset",
    "save_path": "${WORK_DIR}",
    "cpu": false,
    "use_cosine_lr": true,
    "warmup_iters": 1000,
    "min_lr_ratio": 0.01,
    "use_pin_memory": false
}
EOF
        
        # 运行训练脚本并记录日志
        {
            echo "===== CircuitNet 1.0 (N28)数据集训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: ibUNet (Inception Boosted U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $N28_DATA_PATH"
            echo "训练数据文件: ${FILES_PATH}/train_N28.csv"
            echo "使用ibUNet原项目配置参数"
            echo "====================================="
            echo ""
            
            # 添加数据集文件检查脚本
            echo "执行数据集检查..."
            # 使用-u参数禁用Python输出缓冲
            python -u << EOF
import os
import pandas as pd
import numpy as np
import sys

# 强制不缓冲输出
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

csv_path = "${FILES_PATH}/train_N28.csv"
data_root = "${N28_DATA_PATH}"

# 读取CSV文件
df = pd.read_csv(csv_path, header=None)
if len(df.columns) >= 2:
    feature_paths = df[0].tolist()
    label_paths = df[1].tolist()
    print(f"CSV文件包含 {len(feature_paths)} 条记录")
    
    # 检查文件是否存在
    missing_features = 0
    missing_labels = 0
    
    for i, (feat, label) in enumerate(zip(feature_paths, label_paths)):
        feat_path = os.path.join(data_root, feat)
        label_path = os.path.join(data_root, label)
        
        if not os.path.exists(feat_path):
            missing_features += 1
            if missing_features <= 5:
                print(f"特征文件不存在: {feat_path}")
        
        if not os.path.exists(label_path):
            missing_labels += 1
            if missing_labels <= 5:
                print(f"标签文件不存在: {label_path}")
                
        if i < 3:
            try:
                feature_data = np.load(feat_path)
                label_data = np.load(label_path)
                print(f"示例 #{i+1}:")
                print(f"  - 特征形状: {feature_data.shape}")
                print(f"  - 标签形状: {label_data.shape}")
                # 刷新输出
                sys.stdout.flush()
            except Exception as e:
                print(f"无法加载示例 #{i+1}: {e}")
    
    if missing_features > 0:
        print(f"警告: {missing_features} 个特征文件不存在")
    if missing_labels > 0:
        print(f"警告: {missing_labels} 个标签文件不存在")
else:
    print(f"错误: CSV格式不正确，列数: {len(df.columns)}")
EOF
            
            echo ""
            echo "开始训练..."
            python "$SCRIPT_DIR/train_simple.py" \
                --arg_file "$PARAMS_FILE"
                
            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } > >(tee -a "$LOG_FILE") 2>&1

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    train_ispd)
        WORK_DIR="${BASE_WORK_DIR}/congestion_ibunet_ispd"
        mkdir -p "$WORK_DIR"
        LOG_FILE="${LOG_DIR}/ibunet_train_ispd_${TIMESTAMP}.log"

        echo "开始在ISPD2015数据集上训练ibUNet模型..."
        echo "日志将保存到: $LOG_FILE"

        # 检查数据路径
        check_path "$ISPD_DATA_PATH" "ISPD数据集" || exit 1
        check_csv "${FILES_PATH}/train_ISPD2015.csv" "ISPD训练数据索引" || exit 1
        echo "ISPD特征目录: $(ls -l ${ISPD_DATA_PATH}/feature | wc -l) 个文件"
        echo "ISPD标签目录: $(ls -l ${ISPD_DATA_PATH}/label | wc -l) 个文件"

        # 创建临时参数文件 - 与ibUNet原项目完全一致的配置
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "max_iters": 20000,
    "batch_size": 16,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "dataroot": "${ISPD_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_ISPD2015.csv",
    "ann_file_test": "${FILES_PATH}/test_ISPD2015.csv",
    "model_type": "Congestion_Prediction_Net",
    "in_channels": 6,
    "out_channels": 1,
    "loss_type": "MSELoss",
    "aug_pipeline": ["Flip"],
    "dataset_type": "CongestionDataset",
    "save_path": "${WORK_DIR}",
    "cpu": false,
    "use_cosine_lr": true,
    "warmup_iters": 1000,
    "min_lr_ratio": 0.01,
    "use_pin_memory": false
}
EOF

        # 运行统一的训练脚本并记录日志
        {
            echo "===== ISPD2015数据集训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: ibUNet (Inception Boosted U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $ISPD_DATA_PATH"
            echo "使用ibUNet原项目配置参数"
            echo "====================================="
            echo ""

            python "$SCRIPT_DIR/train_simple.py" \
                --arg_file "$PARAMS_FILE"

            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee -a "$LOG_FILE"

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    test_n14)
        WORK_DIR="${BASE_WORK_DIR}/congestion_ibunet"
        LOG_FILE="${LOG_DIR}/ibunet_test_n14_${TIMESTAMP}.log"
        
        echo "测试ibUNet模型（在CircuitNet 2.0 (N14)数据集上）..."
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
    "model_type": "Congestion_Prediction_Net",
    "in_channels": 3,
    "out_channels": 1,
    "load_state_dict": true,
    "dataroot": "${N14_DATA_PATH}",
    "ann_file_test": "${FILES_PATH}/test_N14.csv",
    "dataset_type": "CongestionDataset",
    "cpu": false,
    "eval_metric": ["NRMS", "SSIM", "PSNR", "peak_nrms"]
}
EOF
        
        # 运行测试脚本并记录日志
        {
            echo "===== CircuitNet 2.0 (N14)数据集测试日志 ====="
            echo "开始时间: $(date)"
            echo "模型: ibUNet (Inception Boosted U-Net)"
            echo "检查点: $LATEST_CHECKPOINT"
            echo "数据集路径: $N14_DATA_PATH"
            echo "测试数据文件: ${FILES_PATH}/test_N14.csv"
            echo "====================================="
            echo ""
            
            python "$SCRIPT_DIR/test_simple.py" \
                --arg_file "$PARAMS_FILE" 2>&1
                
            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee -a "$LOG_FILE"

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    test_n28)
        WORK_DIR="${BASE_WORK_DIR}/congestion_ibunet_n28"
        LOG_FILE="${LOG_DIR}/ibunet_test_n28_${TIMESTAMP}.log"
        
        echo "测试ibUNet模型（在CircuitNet 1.0 (N28)数据集上）..."
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
    "model_type": "Congestion_Prediction_Net",
    "in_channels": 3,
    "out_channels": 1,
    "load_state_dict": true,
    "dataroot": "${N28_DATA_PATH}",
    "ann_file_test": "${FILES_PATH}/test_N28.csv",
    "dataset_type": "CongestionDataset",
    "cpu": false,
    "eval_metric": ["NRMS", "SSIM", "PSNR", "peak_nrms"]
}
EOF
        
        # 运行测试脚本并记录日志
        {
            echo "===== CircuitNet 1.0 (N28)数据集测试日志 ====="
            echo "开始时间: $(date)"
            echo "模型: ibUNet (Inception Boosted U-Net)"
            echo "检查点: $LATEST_CHECKPOINT"
            echo "数据集路径: $N28_DATA_PATH"
            echo "测试数据文件: ${FILES_PATH}/test_N28.csv"
            echo "====================================="
            echo ""
            
            python "$SCRIPT_DIR/test_simple.py" \
                --arg_file "$PARAMS_FILE" 2>&1
                
            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee -a "$LOG_FILE"

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    test_ispd)
        WORK_DIR="${BASE_WORK_DIR}/congestion_ibunet_ispd"
        LOG_FILE="${LOG_DIR}/ibunet_test_ispd_${TIMESTAMP}.log"
        
        echo "测试ibUNet模型（在ISPD2015数据集上）..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$ISPD_DATA_PATH" "ISPD数据集" || exit 1
        check_csv "${FILES_PATH}/test_ISPD2015.csv" "ISPD测试数据索引" || exit 1
        
        # 按优先级查找模型文件
        MODEL_PATH=""
        
        # 1. 首先检查是否有best_model.pth
        if [ -f "$WORK_DIR/best_model.pth" ]; then
            MODEL_PATH="$WORK_DIR/best_model.pth"
            echo "使用最佳模型: $MODEL_PATH"
        # 2. 然后检查是否有final_model.pth
        elif [ -f "$WORK_DIR/final_model.pth" ]; then
            MODEL_PATH="$WORK_DIR/final_model.pth"
            echo "使用最终模型: $MODEL_PATH"
        # 3. 如果都没有，使用最新的检查点
        else
            LATEST_CHECKPOINT=$(find "$WORK_DIR" -name "*.pth" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
            
            if [ -z "$LATEST_CHECKPOINT" ]; then
                echo "错误: 未找到模型文件，请先训练模型"
                exit 1
            fi
            
            MODEL_PATH="$LATEST_CHECKPOINT"
            echo "使用最新检查点: $MODEL_PATH"
        fi
        
        # 运行新的ISPD2015测试脚本并记录日志
        {
            echo "===== ISPD2015数据集测试日志 ====="
            echo "开始时间: $(date)"
            echo "模型: ibUNet (Inception Boosted U-Net)"
            echo "模型文件: $MODEL_PATH"
            echo "数据集路径: $ISPD_DATA_PATH"
            echo "====================================="
            echo ""
            
            # 创建临时参数文件用于test_simple.py
            PARAMS_FILE=$(mktemp)
            cat > "$PARAMS_FILE" << EOF
{
    "test_mode": true,
    "pretrained": "$MODEL_PATH",
    "model_type": "Congestion_Prediction_Net",
    "in_channels": 6,
    "out_channels": 1,
    "load_state_dict": true,
    "dataroot": "${ISPD_DATA_PATH}",
    "ann_file_test": "${FILES_PATH}/test_ISPD2015.csv",
    "save_path": "${WORK_DIR}/results",
    "cpu": false,
    "eval_metric": ["NRMS", "SSIM", "PSNR", "peak_nrms"]
}
EOF

            python "$SCRIPT_DIR/test_simple.py" \
                --arg_file "$PARAMS_FILE" 2>&1

            # 删除临时参数文件
            rm "$PARAMS_FILE"
                
            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } | tee -a "$LOG_FILE"
        ;;
    *)
        echo "错误: 未知参数 '$1'"
        echo "用法: $0 [train_n14|train_n28|train_ispd|test_n14|test_n28|test_ispd]"
        exit 1
        ;;
esac

echo "操作完成!"
exit 0 