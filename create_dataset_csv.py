import glob
import os
import random

# 数据目录
feature_dir = '/media/user/6E55DE7F3C46090D2/gky/CircuitNet6/CircuitNet-main/routability_ir_drop_prediction/training_set/congestion/feature'
label_dir = '/media/user/6E55DE7F3C46090D2/gky/CircuitNet6/CircuitNet-main/routability_ir_drop_prediction/training_set/congestion/label'

# 输出CSV文件路径
output_dir = '/media/user/6E55DE7F3C46090D2/gky/CircuitNet6/n14_processed/files'
train_csv = os.path.join(output_dir, 'train_N14.csv')
test_csv = os.path.join(output_dir, 'test_N14.csv')

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 查找特征文件并匹配对应的标签文件
feature_files = glob.glob(os.path.join(feature_dir, '*.npy'))
valid_pairs = []

for feature_file in feature_files:
    basename = os.path.basename(feature_file)
    label_file = os.path.join(label_dir, basename)
    if os.path.exists(label_file):
        valid_pairs.append((feature_file, label_file))

print(f'找到 {len(valid_pairs)} 个有效文件对')

# 划分训练集和测试集（8:2）
random.seed(42)  # 设置随机种子以确保可重复性
random.shuffle(valid_pairs)
split = int(0.8 * len(valid_pairs))
train_pairs = valid_pairs[:split]
test_pairs = valid_pairs[split:]

print(f'训练集: {len(train_pairs)}, 测试集: {len(test_pairs)}')

# 生成训练集CSV文件
with open(train_csv, 'w') as f:
    for feature, label in train_pairs:
        f.write(f'{feature},{label}\n')

# 生成测试集CSV文件
with open(test_csv, 'w') as f:
    for feature, label in test_pairs:
        f.write(f'{feature},{label}\n')

print(f'已生成CSV文件: {train_csv} 和 {test_csv}') 
import os
import random

# 数据目录
feature_dir = '/media/user/6E55DE7F3C46090D2/gky/CircuitNet6/CircuitNet-main/routability_ir_drop_prediction/training_set/congestion/feature'
label_dir = '/media/user/6E55DE7F3C46090D2/gky/CircuitNet6/CircuitNet-main/routability_ir_drop_prediction/training_set/congestion/label'

# 输出CSV文件路径
output_dir = '/media/user/6E55DE7F3C46090D2/gky/CircuitNet6/n14_processed/files'
train_csv = os.path.join(output_dir, 'train_N14.csv')
test_csv = os.path.join(output_dir, 'test_N14.csv')

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 查找特征文件并匹配对应的标签文件
feature_files = glob.glob(os.path.join(feature_dir, '*.npy'))
valid_pairs = []

for feature_file in feature_files:
    basename = os.path.basename(feature_file)
    label_file = os.path.join(label_dir, basename)
    if os.path.exists(label_file):
        valid_pairs.append((feature_file, label_file))

print(f'找到 {len(valid_pairs)} 个有效文件对')

# 划分训练集和测试集（8:2）
random.seed(42)  # 设置随机种子以确保可重复性
random.shuffle(valid_pairs)
split = int(0.8 * len(valid_pairs))
train_pairs = valid_pairs[:split]
test_pairs = valid_pairs[split:]

print(f'训练集: {len(train_pairs)}, 测试集: {len(test_pairs)}')

# 生成训练集CSV文件
with open(train_csv, 'w') as f:
    for feature, label in train_pairs:
        f.write(f'{feature},{label}\n')

# 生成测试集CSV文件
with open(test_csv, 'w') as f:
    for feature, label in test_pairs:
        f.write(f'{feature},{label}\n')

print(f'已生成CSV文件: {train_csv} 和 {test_csv}') 