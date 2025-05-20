import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import dgl
from torch.utils.data import DataLoader
from congestion_gcn_model import build_model
from train_congestion_gnn import GraphDataset, collate_fn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("CongestionTester")


def load_model(
    model_path, model_type="sage", in_dim=12, hidden_dim=128, num_layers=3, device="cpu"
):
    """加载模型"""
    model = build_model(
        model_type=model_type,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=2,
        num_layers=num_layers,
    )

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def predict(model, loader, device, save_dir=None):
    """使用模型进行预测"""
    all_preds = []
    all_labels = []
    all_design_names = []

    with torch.no_grad():
        for bg, design_names in tqdm(loader, desc="Prediction"):
            bg = bg.to(device)
            features = bg.ndata["feat"]
            edge_weights = bg.edata.get("weight", None)

            # 如果有真实标签，获取它们
            if "labels" in bg.ndata:
                labels = bg.ndata["labels"]
                all_labels.append(labels.cpu().numpy())

            # 执行预测
            if edge_weights is not None:
                pred = model(bg, features, edge_weights)
            else:
                pred = model(bg, features)

            # 保存预测结果
            all_preds.append(pred.cpu().numpy())
            all_design_names.extend(design_names)

    # 合并结果
    if all_labels:
        all_labels = np.concatenate(all_labels, axis=0)

    # 获取节点和预测结果之间的映射
    design_to_preds = {}
    node_offset = 0

    for i, design_name in enumerate(all_design_names):
        graph_path = os.path.join(
            loader.dataset.graph_dir, f"{design_name}_congestion.dgl"
        )
        g, _ = dgl.load_graphs(graph_path)
        g = g[0]
        num_nodes = g.num_nodes()

        # 从预测结果中取出对应设计的部分
        if node_offset < len(all_preds[0]):
            design_to_preds[design_name] = all_preds[0][
                node_offset : node_offset + num_nodes
            ]
            node_offset += num_nodes

    # 如果指定了保存目录，保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        for design_name, preds in design_to_preds.items():
            # 保存节点预测结果
            np.save(os.path.join(save_dir, f"{design_name}_node_preds.npy"), preds)

            # 在这里可以将节点预测结果映射回拥塞度地图
            # 需要根据具体应用调整

            # 可视化预测结果（以水平拥塞度为例）
            try:
                # 加载映射文件以获取节点位置信息
                mapping_path = os.path.join(
                    loader.dataset.graph_dir, f"{design_name}_mapping.pkl"
                )
                if os.path.exists(mapping_path):
                    import pickle

                    with open(mapping_path, "rb") as f:
                        mapping = pickle.load(f)

                    # 加载图以获取节点特征（包含位置信息）
                    g, _ = dgl.load_graphs(
                        os.path.join(
                            loader.dataset.graph_dir, f"{design_name}_congestion.dgl"
                        )
                    )
                    g = g[0]
                    node_features = g.ndata["feat"].cpu().numpy()

                    # 创建散点图可视化
                    plt.figure(figsize=(12, 10))

                    # 为每个节点绘制散点，颜色表示预测的水平拥塞度
                    x_coords = node_features[:, 0]  # 中心x坐标
                    y_coords = node_features[:, 1]  # 中心y坐标
                    h_congestion = preds[:, 0]  # 水平拥塞度预测

                    plt.scatter(
                        x_coords, y_coords, c=h_congestion, cmap="hot", alpha=0.7, s=10
                    )
                    plt.colorbar(label="Predicted Horizontal Congestion")
                    plt.title(f"Predicted Horizontal Congestion for {design_name}")
                    plt.xlabel("X Coordinate")
                    plt.ylabel("Y Coordinate")
                    plt.savefig(
                        os.path.join(save_dir, f"{design_name}_h_congestion_pred.png"),
                        dpi=300,
                    )
                    plt.close()

                    # 为垂直拥塞度做同样的可视化
                    plt.figure(figsize=(12, 10))
                    v_congestion = preds[:, 1]  # 垂直拥塞度预测
                    plt.scatter(
                        x_coords, y_coords, c=v_congestion, cmap="hot", alpha=0.7, s=10
                    )
                    plt.colorbar(label="Predicted Vertical Congestion")
                    plt.title(f"Predicted Vertical Congestion for {design_name}")
                    plt.xlabel("X Coordinate")
                    plt.ylabel("Y Coordinate")
                    plt.savefig(
                        os.path.join(save_dir, f"{design_name}_v_congestion_pred.png"),
                        dpi=300,
                    )
                    plt.close()
            except Exception as e:
                logger.error(f"可视化设计 {design_name} 的预测结果时出错: {e}")

    return all_preds, all_labels, design_to_preds


def evaluate(predictions, labels):
    """评估预测结果"""
    if len(predictions) == 0 or len(labels) == 0:
        logger.warning("没有预测结果或标签，无法评估")
        return {}

    # 计算水平和垂直拥塞度的评估指标
    mse_h = mean_squared_error(labels[:, 0], predictions[:, 0])
    mse_v = mean_squared_error(labels[:, 1], predictions[:, 1])
    r2_h = r2_score(labels[:, 0], predictions[:, 0])
    r2_v = r2_score(labels[:, 1], predictions[:, 1])

    metrics = {
        "mse_h": mse_h,
        "mse_v": mse_v,
        "r2_h": r2_h,
        "r2_v": r2_v,
        "avg_mse": (mse_h + mse_v) / 2,
        "avg_r2": (r2_h + r2_v) / 2,
    }

    return metrics


def convert_node_to_gcell_congestion(
    node_preds, node_features, design_info, gcell_size=1
):
    """将节点预测转换为GCell拥塞度地图

    参数:
        node_preds: 节点预测结果 [N, 2]，水平和垂直拥塞度
        node_features: 节点特征 [N, D]，包含节点位置信息
        design_info: 设计信息，包含芯片尺寸
        gcell_size: GCell大小

    返回:
        h_congestion: 水平拥塞度地图
        v_congestion: 垂直拥塞度地图
    """
    # 从设计信息中获取芯片尺寸
    chip_width = design_info.get("width", 1000)
    chip_height = design_info.get("height", 1000)

    # 计算GCell数量
    num_gcell_x = int(np.ceil(chip_width / gcell_size))
    num_gcell_y = int(np.ceil(chip_height / gcell_size))

    # 初始化拥塞度地图
    h_congestion = np.zeros((num_gcell_y, num_gcell_x))
    v_congestion = np.zeros((num_gcell_y, num_gcell_x))
    gcell_count = np.zeros((num_gcell_y, num_gcell_x))

    # 将节点预测映射到GCell
    for i in range(len(node_preds)):
        # 节点中心坐标
        x_center = node_features[i, 0]
        y_center = node_features[i, 1]

        # 计算所在GCell
        gcell_x = min(int(x_center / gcell_size), num_gcell_x - 1)
        gcell_y = min(int(y_center / gcell_size), num_gcell_y - 1)

        # 累加拥塞度
        h_congestion[gcell_y, gcell_x] += node_preds[i, 0]
        v_congestion[gcell_y, gcell_x] += node_preds[i, 1]
        gcell_count[gcell_y, gcell_x] += 1

    # 计算平均拥塞度
    mask = gcell_count > 0
    h_congestion[mask] /= gcell_count[mask]
    v_congestion[mask] /= gcell_count[mask]

    return h_congestion, v_congestion


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="拥塞预测图神经网络测试脚本")

    # 数据参数
    parser.add_argument(
        "--graph_dir", type=str, default="./congestion_graphs", help="图文件目录"
    )
    parser.add_argument(
        "--label_dir", type=str, default=None, help="标签目录（如果与图目录不同）"
    )
    parser.add_argument(
        "--designs_list", type=str, default="./test_designs.csv", help="测试设计名列表"
    )

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument(
        "--model_type",
        type=str,
        default="sage",
        choices=["gcn", "gat", "sage"],
        help="模型类型",
    )
    parser.add_argument("--in_dim", type=int, default=12, help="输入特征维度")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=3, help="图卷积层数")

    # 测试参数
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./work_dir/congestion_gnn/predictions",
        help="保存预测结果的目录",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="不使用CUDA"
    )

    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置设备
    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    logger.info(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    model = load_model(
        args.model_path,
        model_type=args.model_type,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
    )

    # 读取测试设计列表
    if os.path.exists(args.designs_list):
        with open(args.designs_list, "r") as f:
            design_list = [line.strip() for line in f.readlines()]
    else:
        logger.warning(
            f"设计列表文件不存在: {args.designs_list}，使用图文件目录中的所有图"
        )
        design_list = []
        for file in os.listdir(args.graph_dir):
            if file.endswith("_congestion.dgl"):
                design_name = file.replace("_congestion.dgl", "")
                design_list.append(design_name)

    logger.info(f"测试设计数量: {len(design_list)}")

    # 创建测试数据集和数据加载器
    test_dataset = GraphDataset(
        args.graph_dir,
        design_list,
        label_dir=args.label_dir if args.label_dir else args.graph_dir,
        test_mode=False,  # 设为False以加载标签进行评估
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # 预测
    logger.info("开始预测...")
    all_preds, all_labels, design_to_preds = predict(
        model, test_loader, device, args.save_dir
    )

    # 评估
    if len(all_labels) > 0:
        logger.info("评估预测结果...")
        metrics = evaluate(all_preds[0], all_labels)

        # 打印评估结果
        logger.info("评估结果:")
        logger.info(
            f"水平拥塞度 MSE: {metrics['mse_h']:.4f}, R2: {metrics['r2_h']:.4f}"
        )
        logger.info(
            f"垂直拥塞度 MSE: {metrics['mse_v']:.4f}, R2: {metrics['r2_v']:.4f}"
        )
        logger.info(
            f"平均 MSE: {metrics['avg_mse']:.4f}, 平均 R2: {metrics['avg_r2']:.4f}"
        )

        # 保存评估结果
        with open(os.path.join(args.save_dir, "evaluation_results.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        logger.warning("没有标签数据，跳过评估")

    logger.info(f"测试完成，结果保存在 {args.save_dir}")


if __name__ == "__main__":
    main()
