#!/bin/bash

# 脚本名称: run_swinunet.sh
# 功能描述: 运行SwinUNet模型进行拥塞预测任务
# 模型说明: SwinUNet - 基于Swin Transformer的U-Net架构，结合了Transformer的全局建模能力和U-Net的分层特征提取
# 用法示例: 
#   ./run_swinunet.sh train_n14    # 在CircuitNet 2.0 (N14)数据集上训练
#   ./run_swinunet.sh train_n28    # 在CircuitNet 1.0 (N28)数据集上训练
#   ./run_swinunet.sh train_ispd   # 在ISPD2015数据集上训练
#   ./run_swinunet.sh test_n14     # 在CircuitNet 2.0 (N14)数据集上测试
#   ./run_swinunet.sh test_n28     # 在CircuitNet 1.0 (N28)数据集上测试
#   ./run_swinunet.sh test_ispd    # 在ISPD2015数据集上测试

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
        WORK_DIR="${BASE_WORK_DIR}/congestion_swinunet"
        mkdir -p "$WORK_DIR"
        LOG_FILE="${LOG_DIR}/swinunet_train_n14_${TIMESTAMP}.log"
        
        echo "开始在CircuitNet 2.0 (N14)数据集上训练SwinUNet模型..."
        echo "日志将保存到: $LOG_FILE"
        
        # 检查数据路径
        check_path "$N14_DATA_PATH" "N14数据集" || exit 1
        check_csv "${FILES_PATH}/train_N14.csv" "N14训练数据索引" || exit 1
        echo "N14特征目录: $(ls -l ${N14_DATA_PATH}/feature | wc -l) 个文件"
        echo "N14标签目录: $(ls -l ${N14_DATA_PATH}/label | wc -l) 个文件"
        
        # 定义训练参数 - 与ibUNet原项目完全一致的配置
        BATCH_SIZE=8
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
    "model_type": "SwinUNetCongestion",
    "in_channels": 3,
    "out_channels": 1,
    "img_size": 256,
    "patch_size": 4,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "qkv_bias": true,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "ape": false,
    "patch_norm": true,
    "use_checkpoint": false,
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
            echo "模型: SwinUNet (Swin Transformer U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $N14_DATA_PATH"
            echo "训练数据文件: ${FILES_PATH}/train_N14.csv"
            echo "使用SwinUNet标准配置参数"
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
        WORK_DIR="${BASE_WORK_DIR}/congestion_swinunet_n28"
        mkdir -p "$WORK_DIR"
        LOG_FILE="${LOG_DIR}/swinunet_train_n28_${TIMESTAMP}.log"
        
        # 检查外部传入的LOG_FILE变量 - 如果由run_model.sh传入了LOG_FILE，则使用该文件
        if [ -n "$EXTERNAL_LOG_FILE" ]; then
            LOG_FILE="$EXTERNAL_LOG_FILE"
            echo "使用外部指定的日志文件: $LOG_FILE"
        fi
        
        echo "开始在CircuitNet 1.0 (N28)数据集上训练SwinUNet模型..."
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

        # 输出训练信息到终端和日志文件
        {
            echo "===== CircuitNet 1.0 (N28)数据集训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: SwinUNet (Swin Transformer U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $N28_DATA_PATH"
            echo "训练数据文件: ${FILES_PATH}/train_N28.csv"
            echo "使用SwinUNet标准配置参数"
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
    train_ispd)
        WORK_DIR="${BASE_WORK_DIR}/congestion_swinunet_ispd"
        mkdir -p "$WORK_DIR"
        LOG_FILE="${LOG_DIR}/swinunet_train_ispd_${TIMESTAMP}.log"

        echo "开始在ISPD2015数据集上训练SwinUNet模型..."
        echo "日志将保存到: $LOG_FILE"

        # 检查数据路径
        check_path "$ISPD_DATA_PATH" "ISPD2015数据集" || exit 1
        check_csv "${FILES_PATH}/train_ISPD2015.csv" "ISPD2015训练数据索引" || exit 1
        echo "ISPD2015特征目录: $(ls -l ${ISPD_DATA_PATH}/feature | wc -l) 个文件"
        echo "ISPD2015标签目录: $(ls -l ${ISPD_DATA_PATH}/label | wc -l) 个文件"

        # 创建临时参数文件 - ISPD数据集使用6通道输入
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "max_iters": 200000,
    "batch_size": 8,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "dataroot": "${ISPD_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_ISPD2015.csv",
    "ann_file_test": "${FILES_PATH}/test_ISPD2015.csv",
    "model_type": "SwinUNetCongestion",
    "in_channels": 6,
    "out_channels": 1,
    "img_size": 256,
    "patch_size": 4,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "qkv_bias": true,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "ape": false,
    "patch_norm": true,
    "use_checkpoint": false,
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
            echo "===== ISPD2015数据集训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: SwinUNet (Swin Transformer U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $ISPD_DATA_PATH"
            echo "训练数据文件: ${FILES_PATH}/train_ISPD2015.csv"
            echo "输入通道数: 6 (ISPD2015特有)"
            echo "使用SwinUNet标准配置参数"
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
    test_n14)
        WORK_DIR="${BASE_WORK_DIR}/congestion_swinunet"
        LOG_FILE="${LOG_DIR}/swinunet_test_n14_${TIMESTAMP}.log"

        echo "开始在CircuitNet 2.0 (N14)数据集上测试SwinUNet模型..."
        echo "日志将保存到: $LOG_FILE"

        # 检查数据路径
        check_path "$N14_DATA_PATH" "N14数据集" || exit 1
        check_csv "${FILES_PATH}/test_N14.csv" "N14测试数据索引" || exit 1

        # 查找最新的checkpoint
        CHECKPOINT_PATH=$(find "$WORK_DIR" -name "*.pth" -type f | sort -V | tail -1)
        if [ -z "$CHECKPOINT_PATH" ]; then
            echo "错误: 在 $WORK_DIR 中未找到checkpoint文件"
            echo "请先运行训练: $0 train_n14"
            exit 1
        fi
        echo "使用checkpoint: $CHECKPOINT_PATH"

        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "dataroot": "${N14_DATA_PATH}",
    "ann_file_test": "${FILES_PATH}/test_N14.csv",
    "model_type": "SwinUNetCongestion",
    "in_channels": 3,
    "out_channels": 1,
    "img_size": 256,
    "patch_size": 4,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "qkv_bias": true,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "ape": false,
    "patch_norm": true,
    "use_checkpoint": false,
    "dataset_type": "CongestionDataset",
    "checkpoint": "${CHECKPOINT_PATH}",
    "save_path": "${WORK_DIR}/test_results",
    "use_pin_memory": false
}
EOF
{
    "max_iters": 200000,
    "batch_size": 8,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "dataroot": "${N28_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_N28.csv",
    "ann_file_test": "${FILES_PATH}/test_N28.csv",
    "model_type": "SwinUNetCongestion",
    "in_channels": 3,
    "out_channels": 1,
    "img_size": 256,
    "patch_size": 4,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "qkv_bias": true,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "ape": false,
    "patch_norm": true,
    "use_checkpoint": false,
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

        # 输出训练信息到终端和日志文件
        {
            echo "===== CircuitNet 1.0 (N28)数据集训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: SwinUNet (Swin Transformer U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $N28_DATA_PATH"
            echo "训练数据文件: ${FILES_PATH}/train_N28.csv"
            echo "使用SwinUNet标准配置参数"
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
    train_ispd)
        WORK_DIR="${BASE_WORK_DIR}/congestion_swinunet_ispd"
        mkdir -p "$WORK_DIR"
        LOG_FILE="${LOG_DIR}/swinunet_train_ispd_${TIMESTAMP}.log"

        echo "开始在ISPD2015数据集上训练SwinUNet模型..."
        echo "日志将保存到: $LOG_FILE"

        # 检查数据路径
        check_path "$ISPD_DATA_PATH" "ISPD2015数据集" || exit 1
        check_csv "${FILES_PATH}/train_ISPD2015.csv" "ISPD2015训练数据索引" || exit 1
        echo "ISPD2015特征目录: $(ls -l ${ISPD_DATA_PATH}/feature | wc -l) 个文件"
        echo "ISPD2015标签目录: $(ls -l ${ISPD_DATA_PATH}/label | wc -l) 个文件"

        # 创建临时参数文件 - ISPD数据集使用6通道输入
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "max_iters": 200000,
    "batch_size": 8,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "dataroot": "${ISPD_DATA_PATH}",
    "ann_file_train": "${FILES_PATH}/train_ISPD2015.csv",
    "ann_file_test": "${FILES_PATH}/test_ISPD2015.csv",
    "model_type": "SwinUNetCongestion",
    "in_channels": 6,
    "out_channels": 1,
    "img_size": 256,
    "patch_size": 4,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "qkv_bias": true,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "ape": false,
    "patch_norm": true,
    "use_checkpoint": false,
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
            echo "===== ISPD2015数据集训练日志 ====="
            echo "开始时间: $(date)"
            echo "模型: SwinUNet (Swin Transformer U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $ISPD_DATA_PATH"
            echo "训练数据文件: ${FILES_PATH}/train_ISPD2015.csv"
            echo "输入通道数: 6 (ISPD2015特有)"
            echo "使用SwinUNet标准配置参数"
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
    test_n14)
        WORK_DIR="${BASE_WORK_DIR}/congestion_swinunet"
        LOG_FILE="${LOG_DIR}/swinunet_test_n14_${TIMESTAMP}.log"

        echo "开始在CircuitNet 2.0 (N14)数据集上测试SwinUNet模型..."
        echo "日志将保存到: $LOG_FILE"

        # 检查数据路径
        check_path "$N14_DATA_PATH" "N14数据集" || exit 1
        check_csv "${FILES_PATH}/test_N14.csv" "N14测试数据索引" || exit 1

        # 查找最新的checkpoint
        CHECKPOINT_PATH=$(find "$WORK_DIR" -name "*.pth" -type f | sort -V | tail -1)
        if [ -z "$CHECKPOINT_PATH" ]; then
            echo "错误: 在 $WORK_DIR 中未找到checkpoint文件"
            echo "请先运行训练: $0 train_n14"
            exit 1
        fi
        echo "使用checkpoint: $CHECKPOINT_PATH"

        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "dataroot": "${N14_DATA_PATH}",
    "ann_file_test": "${FILES_PATH}/test_N14.csv",
    "model_type": "SwinUNetCongestion",
    "in_channels": 3,
    "out_channels": 1,
    "img_size": 256,
    "patch_size": 4,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "qkv_bias": true,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "ape": false,
    "patch_norm": true,
    "use_checkpoint": false,
    "dataset_type": "CongestionDataset",
    "checkpoint": "${CHECKPOINT_PATH}",
    "save_path": "${WORK_DIR}/test_results",
    "use_pin_memory": false
}
EOF

        # 输出测试信息到终端和日志文件
        {
            echo "===== CircuitNet 2.0 (N14)数据集测试日志 ====="
            echo "开始时间: $(date)"
            echo "模型: SwinUNet (Swin Transformer U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $N14_DATA_PATH"
            echo "测试数据文件: ${FILES_PATH}/test_N14.csv"
            echo "Checkpoint: $CHECKPOINT_PATH"
            echo "====================================="
            echo ""

            # 运行测试脚本
            python "$SCRIPT_DIR/test_simple.py" \
                --arg_file "$PARAMS_FILE"

            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } > >(tee "$LOG_FILE") 2>&1

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    test_n28)
        WORK_DIR="${BASE_WORK_DIR}/congestion_swinunet_n28"
        LOG_FILE="${LOG_DIR}/swinunet_test_n28_${TIMESTAMP}.log"

        echo "开始在CircuitNet 1.0 (N28)数据集上测试SwinUNet模型..."
        echo "日志将保存到: $LOG_FILE"

        # 检查数据路径
        check_path "$N28_DATA_PATH" "N28数据集" || exit 1
        check_csv "${FILES_PATH}/test_N28.csv" "N28测试数据索引" || exit 1

        # 查找最新的checkpoint
        CHECKPOINT_PATH=$(find "$WORK_DIR" -name "*.pth" -type f | sort -V | tail -1)
        if [ -z "$CHECKPOINT_PATH" ]; then
            echo "错误: 在 $WORK_DIR 中未找到checkpoint文件"
            echo "请先运行训练: $0 train_n28"
            exit 1
        fi
        echo "使用checkpoint: $CHECKPOINT_PATH"

        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "dataroot": "${N28_DATA_PATH}",
    "ann_file_test": "${FILES_PATH}/test_N28.csv",
    "model_type": "SwinUNetCongestion",
    "in_channels": 3,
    "out_channels": 1,
    "img_size": 256,
    "patch_size": 4,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "qkv_bias": true,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "ape": false,
    "patch_norm": true,
    "use_checkpoint": false,
    "dataset_type": "CongestionDataset",
    "checkpoint": "${CHECKPOINT_PATH}",
    "save_path": "${WORK_DIR}/test_results",
    "use_pin_memory": false
}
EOF

        # 输出测试信息到终端和日志文件
        {
            echo "===== CircuitNet 1.0 (N28)数据集测试日志 ====="
            echo "开始时间: $(date)"
            echo "模型: SwinUNet (Swin Transformer U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $N28_DATA_PATH"
            echo "测试数据文件: ${FILES_PATH}/test_N28.csv"
            echo "Checkpoint: $CHECKPOINT_PATH"
            echo "====================================="
            echo ""

            # 运行测试脚本
            python "$SCRIPT_DIR/test_simple.py" \
                --arg_file "$PARAMS_FILE"

            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } > >(tee "$LOG_FILE") 2>&1

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    test_ispd)
        WORK_DIR="${BASE_WORK_DIR}/congestion_swinunet_ispd"
        LOG_FILE="${LOG_DIR}/swinunet_test_ispd_${TIMESTAMP}.log"

        echo "开始在ISPD2015数据集上测试SwinUNet模型..."
        echo "日志将保存到: $LOG_FILE"

        # 检查数据路径
        check_path "$ISPD_DATA_PATH" "ISPD2015数据集" || exit 1
        check_csv "${FILES_PATH}/test_ISPD2015.csv" "ISPD2015测试数据索引" || exit 1

        # 查找最新的checkpoint
        CHECKPOINT_PATH=$(find "$WORK_DIR" -name "*.pth" -type f | sort -V | tail -1)
        if [ -z "$CHECKPOINT_PATH" ]; then
            echo "错误: 在 $WORK_DIR 中未找到checkpoint文件"
            echo "请先运行训练: $0 train_ispd"
            exit 1
        fi
        echo "使用checkpoint: $CHECKPOINT_PATH"

        # 创建临时参数文件
        PARAMS_FILE=$(mktemp)
        cat > "$PARAMS_FILE" << EOF
{
    "dataroot": "${ISPD_DATA_PATH}",
    "ann_file_test": "${FILES_PATH}/test_ISPD2015.csv",
    "model_type": "SwinUNetCongestion",
    "in_channels": 6,
    "out_channels": 1,
    "img_size": 256,
    "patch_size": 4,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "qkv_bias": true,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "ape": false,
    "patch_norm": true,
    "use_checkpoint": false,
    "dataset_type": "CongestionDataset",
    "checkpoint": "${CHECKPOINT_PATH}",
    "save_path": "${WORK_DIR}/test_results",
    "use_pin_memory": false
}
EOF

        # 输出测试信息到终端和日志文件
        {
            echo "===== ISPD2015数据集测试日志 ====="
            echo "开始时间: $(date)"
            echo "模型: SwinUNet (Swin Transformer U-Net)"
            echo "工作目录: $WORK_DIR"
            echo "数据集路径: $ISPD_DATA_PATH"
            echo "测试数据文件: ${FILES_PATH}/test_ISPD2015.csv"
            echo "输入通道数: 6 (ISPD2015特有)"
            echo "Checkpoint: $CHECKPOINT_PATH"
            echo "====================================="
            echo ""

            # 运行测试脚本
            python "$SCRIPT_DIR/test_simple.py" \
                --arg_file "$PARAMS_FILE"

            echo ""
            echo "====================================="
            echo "结束时间: $(date)"
        } > >(tee "$LOG_FILE") 2>&1

        # 删除临时参数文件
        rm "$PARAMS_FILE"
        ;;
    *)
        echo "错误: 无效的参数 '$1'"
        echo "用法: $0 [train_n14|train_n28|train_ispd|test_n14|test_n28|test_ispd]"
        echo ""
        echo "可用选项:"
        echo "  train_n14  - 在CircuitNet 2.0 (N14)数据集上训练SwinUNet模型"
        echo "  train_n28  - 在CircuitNet 1.0 (N28)数据集上训练SwinUNet模型"
        echo "  train_ispd - 在ISPD2015数据集上训练SwinUNet模型"
        echo "  test_n14   - 在CircuitNet 2.0 (N14)数据集上测试SwinUNet模型"
        echo "  test_n28   - 在CircuitNet 1.0 (N28)数据集上测试SwinUNet模型"
        echo "  test_ispd  - 在ISPD2015数据集上测试SwinUNet模型"
        echo ""
        echo "模型说明: SwinUNet - 基于Swin Transformer的U-Net架构"
        echo "特点: 结合了Transformer的全局建模能力和U-Net的分层特征提取"
        exit 1
        ;;
esac

echo "SwinUNet脚本执行完成!"
