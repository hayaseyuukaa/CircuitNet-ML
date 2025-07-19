# Copyright 2022 CircuitNet. All rights reserved.

from __future__ import print_function
import os
import os.path as osp
import json
import numpy as np
import torch
import seaborn as sns
import matplotlib

matplotlib.use("Agg")  # 使用非GUI后端
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import Parser


def test():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)

    # 加载配置文件
    if arg.arg_file:
        with open(arg.arg_file, "rt") as f:
            arg_dict.update(json.load(f))

    # 初始化路径参数
    arg_dict.setdefault("ann_file_test", "./files/test_N14.csv")
    arg_dict["ann_file"] = arg_dict["ann_file_test"]
    arg_dict["test_mode"] = True
    arg_dict.setdefault("plot_heatmap", False)  # 新增热力图开关

    print("===> Loading datasets")
    dataset = build_dataset(arg_dict)

    print("===> Building model")
    model = build_model(arg_dict)
    if not arg_dict["cpu"]:
        model = model.cuda()

    # ================== 指标系统初始化 ==================
    metrics = {}
    avg_metrics = {}

    for metric_name in arg_dict["eval_metric"]:
        if metric_name == "peak_nrms":

            def wrapped_peak_nrmse(target, pred):
                raw_dict = build_metric(metric_name)(target, pred)
                return {
                    "main": np.mean(list(raw_dict.values())),
                    **{f"peak_{p}": v for p, v in raw_dict.items()},
                }

            metric_func = wrapped_peak_nrmse
        else:
            metric_func = build_metric(metric_name)

        sample_result = metric_func(torch.rand(1, 256, 256), torch.rand(1, 256, 256))
        avg_metrics[metric_name] = (
            {k: 0.0 for k in sample_result.keys()}
            if isinstance(sample_result, dict)
            else {"main": 0.0}
        )
        metrics[metric_name] = metric_func

    # ================== 推理主循环 ==================
    with tqdm(total=len(dataset)) as bar:
        for feature, label, label_path in dataset:
            device = "cuda" if not arg_dict["cpu"] else "cpu"
            input_tensor = feature.to(device)
            target_tensor = label.to(device)

            with torch.no_grad():
                prediction = model(input_tensor)

            # 数据转换
            target_cpu = target_tensor.cpu().squeeze()
            pred_cpu = prediction.squeeze().cpu()

            # 指标计算
            for metric_name, metric_func in metrics.items():
                result = metric_func(target_cpu, pred_cpu)
                result = result if isinstance(result, dict) else {"main": result}
                for sub_name, value in result.items():
                    avg_metrics[metric_name][sub_name] += value

            if arg_dict.get("plot_heatmap", False):
                # 创建保存路径
                heatmap_dir = osp.join(
                    arg_dict.get("save_path", "results"), "congestion_heatmaps"
                )
                os.makedirs(heatmap_dir, exist_ok=True)

                # 获取基础文件名（根据实际路径结构调整）
                base_name = osp.splitext(osp.basename(str(label_path)))[
                    0
                ]  # 确保转换为字符串

                # 创建对比图
                plt.figure(figsize=(18, 6))

                # 预测热图（单通道）
                plt.subplot(1, 3, 1)
                sns.heatmap(
                    pred_cpu.squeeze().numpy(),
                    cmap="hot",
                    vmin=0,
                    vmax=1,
                    cbar_kws={"label": "Congestion Level"},
                )
                plt.title("Predicted Congestion")
                plt.axis("off")

                # 真实热图（单通道）
                plt.subplot(1, 3, 2)
                sns.heatmap(
                    target_cpu.squeeze().numpy(),
                    cmap="hot",
                    vmin=0,
                    vmax=1,
                    cbar_kws={"label": "Congestion Level"},
                )
                plt.title("Ground Truth Congestion")
                plt.axis("off")

                # 保存和清理
                plt.tight_layout()
                plt.savefig(
                    osp.join(heatmap_dir, f"{base_name}_congestion.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            # 原始数据保存
            if arg_dict.get("plot_roc", False):
                save_dir = osp.join(arg_dict.get("save_path", "results"), "test_result")
                os.makedirs(save_dir, exist_ok=True)
                np.save(osp.join(save_dir, f"{base_name}.npy"), pred_cpu.numpy())

            bar.update(1)

    # ================== 结果输出 ==================
    print("\n=== Final Metrics ===")
    for metric_name, sub_metrics in avg_metrics.items():
        print(f"** {metric_name.upper()} **")

        if metric_name == "peak_nrms":
            # 主指标输出
            main_avg = sub_metrics["main"] / len(dataset)
            print(f"  Overall Average: {main_avg:.4f}")

            # 提取并处理百分比指标（关键修复）
            peak_items = []
            for key in sub_metrics:
                if key.startswith("peak_"):
                    # 使用字符串切片提取数值部分
                    percent = float(key[5:])  # 从'peak_0.5'取出0.5
                    peak_items.append((percent, sub_metrics[key]))

            # 按百分比数值排序
            sorted_items = sorted(peak_items, key=lambda x: x)

            # 输出结果
            total = 0.0
            for percent, val in sorted_items:
                avg_val = val / len(dataset)
                print(f"  Top {percent}%: {avg_val:.4f}")
                total += avg_val

        else:
            for sub_name, total in sub_metrics.items():
                avg_val = total / len(dataset)
                print(f"  {sub_name}: {avg_val:.4f}")

    # ROC/PRC处理
    if arg_dict.get("plot_roc", False):
        roc_auc, prc_auc = build_roc_prc_metric(**arg_dict)
        print(f"\nROC AUC: {roc_auc:.4f}")
        print(f"PRC AUC: {prc_auc:.4f}")


if __name__ == "__main__":
    test()
