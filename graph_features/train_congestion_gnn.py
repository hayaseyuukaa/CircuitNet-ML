import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
# 修改DataLoader导入
try:
    from torch.utils.data import DataLoader
except ImportError:
    from torch.utils.data.dataloader import DataLoader

import dgl
from congestion_gcn_model import build_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("CongestionTrainer")


class GraphDataset(Dataset):
    """图数据集类"""

    def __init__(self, graph_dir, design_list, label_dir=None, test_mode=False):
        """
        参数:
            graph_dir: 图文件目录
            design_list: 设计名列表
            label_dir: 标签目录（如果与图目录不同）
            test_mode: 是否为测试模式（无标签）
        """
        self.graph_dir = graph_dir
        self.label_dir = label_dir if label_dir else graph_dir
        self.design_list = design_list
        self.test_mode = test_mode

        # 加载所有图文件路径
        self.graph_paths = []
        for design in design_list:
            graph_path = os.path.join(graph_dir, f"{design}_congestion.dgl")
            if os.path.exists(graph_path):
                self.graph_paths.append(graph_path)
            else:
                logger.warning(f"找不到图文件: {graph_path}")

        logger.info(f"加载了 {len(self.graph_paths)} 个有效图文件")

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        graph_path = self.graph_paths[idx]
        design_name = os.path.basename(graph_path).replace("_congestion.dgl", "")

        # 加载图
        g, _ = dgl.load_graphs(graph_path)
        g = g[0]

        # 如果不是测试模式，还需要加载真实拥塞度标签
        if not self.test_mode:
            # 从拥塞度地图加载真实标签
            congestion_path = os.path.join(
                self.label_dir, f"{design_name}_congestion.npy"
            )
            congestion_map = np.load(congestion_path, allow_pickle=True).item()
            h_congestion = congestion_map["horizontal"]
            v_congestion = congestion_map["vertical"]

            # 将标签添加到图中
            # 注意：由于我们是针对每个节点预测拥塞度
            # 所以需要将全局拥塞度转换为每个节点的局部拥塞度
            # 这已经在构建图时完成，标签存储在节点特征的最后两个维度
            node_features = g.ndata["feat"]
            if node_features.shape[1] >= 12:  # 确保特征包含拥塞度
                true_congestion = node_features[:, -2:]  # 最后两个特征是拥塞度
                g.ndata["labels"] = true_congestion
            else:
                # 如果节点特征不包含拥塞度，创建零标签
                g.ndata["labels"] = torch.zeros((g.num_nodes(), 2), dtype=torch.float32)
                logger.warning(f"设计 {design_name} 的节点特征不包含拥塞度信息")

        return g, design_name


def collate_fn(batch):
    """批处理函数"""
    graphs, design_names = zip(*batch)
    batched_graph = dgl.batch(graphs)
    return batched_graph, design_names


def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (bg, _) in enumerate(tqdm(loader, desc="Training")):
        bg = bg.to(device)
        features = bg.ndata["feat"]
        edge_weights = bg.edata.get("weight", None)
        labels = bg.ndata["labels"]

        optimizer.zero_grad()

        if edge_weights is not None:
            pred = model(bg, features, edge_weights)
        else:
            pred = model(bg, features)

        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # 如果没有验证数据，返回默认值
    if len(loader.dataset) == 0:
        return {
            "loss": 0.0,
            "mse_h": 0.0,
            "mse_v": 0.0,
            "r2_h": 0.0,
            "r2_v": 0.0,
        }

    with torch.no_grad():
        for bg, _ in tqdm(loader, desc="Validation"):
            bg = bg.to(device)
            features = bg.ndata["feat"]
            edge_weights = bg.edata.get("weight", None)
            labels = bg.ndata["labels"]

            if edge_weights is not None:
                pred = model(bg, features, edge_weights)
            else:
                pred = model(bg, features)

            loss = criterion(pred, labels)
            total_loss += loss.item()

            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 如果有预测值，计算评估指标
    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # 计算评估指标
        mse_h = mean_squared_error(all_labels[:, 0], all_preds[:, 0])
        mse_v = mean_squared_error(all_labels[:, 1], all_preds[:, 1])
        r2_h = r2_score(all_labels[:, 0], all_preds[:, 0])
        r2_v = r2_score(all_labels[:, 1], all_preds[:, 1])
    else:
        mse_h = mse_v = r2_h = r2_v = 0.0

    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "mse_h": mse_h,
        "mse_v": mse_v,
        "r2_h": r2_h,
        "r2_v": r2_v,
    }

    return metrics


def predict(model, loader, device, save_dir=None):
    """模型预测"""
    model.eval()
    all_preds = []
    all_design_names = []

    with torch.no_grad():
        for bg, design_names in tqdm(loader, desc="Prediction"):
            bg = bg.to(device)
            features = bg.ndata["feat"]
            edge_weights = bg.edata.get("weight", None)

            if edge_weights is not None:
                pred = model(bg, features, edge_weights)
            else:
                pred = model(bg, features)

            all_preds.append(pred.cpu().numpy())
            all_design_names.extend(design_names)

    # 保存预测结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 当前为批处理后的结果，需要按设计名分离
        start_idx = 0
        for i, design_name in enumerate(all_design_names):
            g, _ = dgl.load_graphs(
                os.path.join(loader.dataset.graph_dir, f"{design_name}_congestion.dgl")
            )
            g = g[0]
            num_nodes = g.num_nodes()

            # 从当前批次中提取对应设计的预测结果
            design_preds = all_preds[start_idx : start_idx + num_nodes]
            start_idx += num_nodes

            # 构建拥塞度地图并保存
            # 这里简化处理，实际应用中可能需要将节点预测结果映射回拥塞度地图
            np.save(
                os.path.join(save_dir, f"{design_name}_pred_congestion.npy"),
                design_preds,
            )


def plot_training_history(history, save_path):
    """绘制训练历史"""
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # MSE曲线
    plt.subplot(2, 2, 2)
    plt.plot(history["val_mse_h"], label="Horizontal MSE")
    plt.plot(history["val_mse_v"], label="Vertical MSE")
    plt.title("Mean Squared Error")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    # R2曲线
    plt.subplot(2, 2, 3)
    plt.plot(history["val_r2_h"], label="Horizontal R2")
    plt.plot(history["val_r2_v"], label="Vertical R2")
    plt.title("R2 Score")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="拥塞预测图神经网络训练脚本")

    # 数据参数
    parser.add_argument(
        "--graph_dir", type=str, default="./congestion_graphs", help="图文件目录"
    )
    parser.add_argument(
        "--label_dir", type=str, default=None, help="标签目录（如果与图目录不同）"
    )
    parser.add_argument(
        "--designs_list", type=str, default="./designs.csv", help="设计名列表"
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")

    # 模型参数
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
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout比率")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="权重衰减")
    parser.add_argument("--early_stopping", type=int, default=20, help="早停轮数")

    # 其他参数
    parser.add_argument(
        "--save_dir", type=str, default="./work_dir/congestion_gnn", help="保存目录"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="不使用CUDA"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练检查点")

    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    logger.info(f"使用设备: {device}")

    # 读取设计名列表
    with open(args.designs_list, "r") as f:
        design_list = [line.strip() for line in f.readlines()]

    # 划分训练集和验证集
    np.random.shuffle(design_list)
    val_size = int(len(design_list) * args.val_ratio)
    train_designs = design_list[val_size:]
    val_designs = design_list[:val_size]

    # 创建数据集和数据加载器
    train_dataset = GraphDataset(args.graph_dir, train_designs, args.label_dir)
    val_dataset = GraphDataset(args.graph_dir, val_designs, args.label_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 构建模型
    model = build_model(
        model_type=args.model_type,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=2,  # 水平和垂直拥塞度
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = model.to(device)

    # 打印模型结构
    logger.info(f"模型结构:\n{model}")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 恢复训练（如果指定检查点）
    start_epoch = 0
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mse_h": [],
        "val_mse_v": [],
        "val_r2_h": [],
        "val_r2_v": [],
    }

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"加载检查点: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            history = checkpoint["history"]
            logger.info(f"恢复训练从第 {start_epoch} 轮开始")
        else:
            logger.warning(f"检查点不存在: {args.resume}")

    # 训练循环
    early_stop_counter = 0

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")

        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_metrics = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step(val_metrics["loss"])

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mse_h"].append(val_metrics["mse_h"])
        history["val_mse_v"].append(val_metrics["mse_v"])
        history["val_r2_h"].append(val_metrics["r2_h"])
        history["val_r2_v"].append(val_metrics["r2_v"])

        # 打印结果
        logger.info(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}"
        )
        logger.info(
            f"Horizontal MSE: {val_metrics['mse_h']:.4f}, R2: {val_metrics['r2_h']:.4f}"
        )
        logger.info(
            f"Vertical MSE: {val_metrics['mse_v']:.4f}, R2: {val_metrics['r2_v']:.4f}"
        )

        # 保存最佳模型
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            early_stop_counter = 0

            # 保存模型
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                os.path.join(args.save_dir, "best_model.pth"),
            )

            logger.info(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            logger.info(
                f"验证损失未改善，早停计数器: {early_stop_counter}/{args.early_stopping}"
            )

        # 早停
        if early_stop_counter >= args.early_stopping:
            logger.info(f"早停触发，停止训练")
            break

        # 每个epoch绘制训练历史
        plot_training_history(
            history, os.path.join(args.save_dir, "training_history.png")
        )

    # 最终评估
    logger.info("加载最佳模型进行最终评估")
    checkpoint = torch.load(os.path.join(args.save_dir, "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = validate(model, val_loader, criterion, device)
    logger.info(f"最终验证结果: Loss={val_metrics['loss']:.4f}")
    logger.info(
        f"Horizontal MSE: {val_metrics['mse_h']:.4f}, R2: {val_metrics['r2_h']:.4f}"
    )
    logger.info(
        f"Vertical MSE: {val_metrics['mse_v']:.4f}, R2: {val_metrics['r2_v']:.4f}"
    )

    # 保存最终结果
    with open(os.path.join(args.save_dir, "final_results.txt"), "w") as f:
        f.write(f"Final Validation Results:\n")
        f.write(f"Loss: {val_metrics['loss']:.4f}\n")
        f.write(
            f"Horizontal MSE: {val_metrics['mse_h']:.4f}, R2: {val_metrics['r2_h']:.4f}\n"
        )
        f.write(
            f"Vertical MSE: {val_metrics['mse_v']:.4f}, R2: {val_metrics['r2_v']:.4f}\n"
        )

    logger.info(f"训练完成，结果保存在 {args.save_dir}")


if __name__ == "__main__":
    main()
