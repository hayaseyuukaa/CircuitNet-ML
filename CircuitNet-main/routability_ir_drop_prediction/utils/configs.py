# Copyright 2022 CircuitNet. All rights reserved.
import argparse
import os
import sys

sys.path.append(os.getcwd())

# 项目根目录路径
PROJECT_ROOT = "/doc/gky/CircuitNet6/CircuitNet-main"
# 数据根目录路径
DATA_ROOT = os.path.join(PROJECT_ROOT, "routability_ir_drop_prediction")
# 训练集路径
TRAINING_SET_PATH = os.path.join(DATA_ROOT, "training_set")
# 文件路径
FILES_PATH = os.path.join(DATA_ROOT, "files")
# 工作目录路径
WORK_DIR = "/doc/gky/CircuitNet6/work_dir"
# 图特征路径
GRAPH_FEATURES_PATH = "/doc/gky/CircuitNet6/graph_features"

# 定义任务配置字典
TASKS = {}

class Parser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--task", default="congestion_routenet")
        self.parser.add_argument(
            "--save_path",
            default=os.path.join(WORK_DIR, "congestion"),
        )
        self.parser.add_argument("--pretrained", default=None)
        self.parser.add_argument("--max_iters", default=200000)
        self.parser.add_argument("--plot_roc", action="store_true")
        self.parser.add_argument("--arg_file", default=None)
        self.parser.add_argument("--cpu", action="store_true")
        self.parser.add_argument("--plot_heatmap", action="store_true")

        self.get_remainder()

    def get_remainder(self):
        args = self.parser.parse_args()
        if args.task == "congestion_gat":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="GATCongestion")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--hidden_channels", default=64)
            self.parser.add_argument("--heads", default=4)
            self.parser.add_argument("--dropout", default=0.6)
            self.parser.add_argument("--lr", default=1e-3)
            self.parser.add_argument("--weight_decay", default=5e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "PeakNRMS"]
            )
        elif args.task == "congestion_visionmamba":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=1, type=int)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="VisionMambaUNet")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--depths", default=[1, 1, 1])
            self.parser.add_argument("--dims", default=[16, 32, 64])
            self.parser.add_argument("--d_state", default=8)
            self.parser.add_argument("--lr", default=2e-4)
            self.parser.add_argument("--weight_decay", default=0)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
        elif args.task == "congestion_swin":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="SwinTransformerCongestion")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--img_size", default=256)
            self.parser.add_argument("--patch_size", default=4)
            self.parser.add_argument("--embed_dim", default=96)
            self.parser.add_argument("--depths", default=[2, 2, 6, 2])
            self.parser.add_argument("--num_heads", default=[3, 6, 12, 24])
            self.parser.add_argument("--window_size", default=8)
            self.parser.add_argument("--mlp_ratio", default=4.0)
            self.parser.add_argument("--drop_rate", default=0.0)
            self.parser.add_argument("--attn_drop_rate", default=0.0)
            self.parser.add_argument("--drop_path_rate", default=0.1)
            self.parser.add_argument("--use_checkpoint", default=False)
            self.parser.add_argument("--lr", default=2e-4)
            self.parser.add_argument("--weight_decay", default=1e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
        elif args.task == "congestion_gpdl":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="GPDL")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--lr", default=2e-4)
            self.parser.add_argument("--weight_decay", default=0)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
        elif args.task == "congestion_routenet":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="RouteNet")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--lr", default=1e-3)
            self.parser.add_argument("--weight_decay", default=5e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
        elif args.task == "congestion_gcn":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="GCNCongestion")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--hidden_channels", default=64)
            self.parser.add_argument("--dropout", default=0.6)
            self.parser.add_argument("--lr", default=1e-3)
            self.parser.add_argument("--weight_decay", default=5e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
        elif args.task == "congestion_graphsage":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="GraphSAGECongestion")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--hidden_channels", default=64)
            self.parser.add_argument("--num_layers", default=2)
            self.parser.add_argument("--dropout", default=0.6)
            self.parser.add_argument("--aggr", default="mean")
            self.parser.add_argument("--lr", default=1e-3)
            self.parser.add_argument("--weight_decay", default=5e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
        elif args.task == "congestion_rgcn":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="RGCNCongestion")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--hidden_channels", default=64)
            self.parser.add_argument("--num_layers", default=2)
            self.parser.add_argument("--num_relations", default=4)
            self.parser.add_argument("--dropout", default=0.6)
            self.parser.add_argument("--num_blocks", default=None)
            self.parser.add_argument("--aggr", default="mean")
            self.parser.add_argument("--lr", default=1e-3)
            self.parser.add_argument("--weight_decay", default=5e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
        elif args.task == "congestion_cgan":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="CGANCongestion")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--noise_dim", default=100)
            self.parser.add_argument("--lr", default=2e-4)
            self.parser.add_argument("--weight_decay", default=0)
            self.parser.add_argument("--loss_type", default="BCEWithLogitsLoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "PeakNRMS"]
            )
        elif args.task == "congestion_fcn":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="FCNCongestion")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--lr", default=1e-3)
            self.parser.add_argument("--weight_decay", default=5e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
        elif args.task == "congestion_lhnn":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=1)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="LHNNCongestion")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--lr", default=1e-4)
            self.parser.add_argument("--weight_decay", default=5e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
        elif args.task == "congestion_gpdl_enhanced":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="GPDL_Enhanced")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--latent_dim", default=64)
            self.parser.add_argument("--lr", default=2e-4)
            self.parser.add_argument("--weight_decay", default=1e-3)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "peak_nrms"]
            )
            self.parser.add_argument("--recon_weight", default=0.5, type=float)
            self.parser.add_argument("--kl_weight", default=0.01, type=float)
        elif args.task == "congestion_cvae":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="CVAECongestion")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--latent_dim", default=100)
            self.parser.add_argument("--lr", default=2e-4)
            self.parser.add_argument("--weight_decay", default=1e-5)
            self.parser.add_argument("--loss_type", default="VAELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "PeakNRMS"]
            )
            self.parser.add_argument("--initial_beta", default=0.01, type=float)
            self.parser.add_argument("--final_beta", default=1.0, type=float)
            self.parser.add_argument("--recon_weight", default=1.0, type=float)
            self.parser.add_argument("--initial_noise", default=0.1, type=float)
            self.parser.add_argument("--final_noise", default=0.001, type=float)
            self.parser.add_argument("--num_samples", default=5, type=int)
        elif args.task == "congestion_gnn":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--graph_dir", 
                default=os.path.join(GRAPH_FEATURES_PATH, "congestion_graphs_N14")
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionGraphDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--model_type", default="CongestionGNN")
            self.parser.add_argument(
                "--gnn_type", default="sage", choices=["gcn", "gat", "sage"]
            )
            self.parser.add_argument("--in_dim", default=12)
            self.parser.add_argument("--hidden_dim", default=128)
            self.parser.add_argument("--out_dim", default=2)
            self.parser.add_argument("--num_layers", default=3)
            self.parser.add_argument("--dropout", default=0.2)
            self.parser.add_argument("--lr", default=0.001)
            self.parser.add_argument("--weight_decay", default=5e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument("--eval-metric", default=["MSE", "R2"])
            self.parser.add_argument("--early_stopping", default=20, type=int)
        elif args.task == "congestion_gpdl_ispd2015":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "ISPD2015_congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_ISPD2015.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_ISPD2015.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="GPDL")
            self.parser.add_argument("--in_channels", default=6)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--lr", default=1.5e-4)
            self.parser.add_argument("--weight_decay", default=5e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
            self.parser.add_argument("--epoch", default=100, type=int)
            self.parser.add_argument("--steps", nargs="+", type=int, default=[40, 70, 90])
            self.parser.add_argument("--n_threads", default=8, type=int)
            self.parser.add_argument("--resume_from", default=None)
            self.parser.add_argument("--save_dir", default=os.path.join(WORK_DIR, 'congestion_gpdl_ispd2015'))
            self.parser.add_argument("--draw_thresh", default=0.5, type=float)
        elif args.task == "congestion_gcn_ispd2015":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "ISPD2015_congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_ISPD2015.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_ISPD2015.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=4)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="GCNCongestion")
            self.parser.add_argument("--in_channels", default=6)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--hidden_channels", default=64)
            self.parser.add_argument("--dropout", default=0.5)
            self.parser.add_argument("--lr", default=1e-4)
            self.parser.add_argument("--weight_decay", default=5e-4)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "peak_nrms"]
            )
            self.parser.add_argument("--n_threads", default=8, type=int)
            self.parser.add_argument("--resume_from", default=None)
            self.parser.add_argument("--save_dir", default=os.path.join(WORK_DIR, 'congestion_gcn_ispd2015'))
            self.parser.add_argument("--draw_thresh", default=0.5, type=float)
        elif args.task == "congestion_visionmamba_v2":
            self.parser.add_argument(
                "--dataroot",
                default=os.path.join(TRAINING_SET_PATH, "congestion"),
            )
            self.parser.add_argument(
                "--ann_file_train", default=os.path.join(FILES_PATH, "train_N14.csv")
            )
            self.parser.add_argument("--ann_file_test", default=os.path.join(FILES_PATH, "test_N14.csv"))
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            self.parser.add_argument("--batch_size", default=1, type=int)
            self.parser.add_argument("--aug_pipeline", default=["Flip"])
            self.parser.add_argument("--model_type", default="VisionMambaUNetV2")
            self.parser.add_argument("--in_channels", default=3)
            self.parser.add_argument("--out_channels", default=1)
            self.parser.add_argument("--depths", default=[2, 2, 2, 2])
            self.parser.add_argument("--dims", default=[32, 64, 128, 256])
            self.parser.add_argument("--d_state", default=16)
            self.parser.add_argument("--deep_supervision", default=False)
            self.parser.add_argument("--lr", default=2e-4)
            self.parser.add_argument("--weight_decay", default=0)
            self.parser.add_argument("--loss_type", default="MSELoss")
            self.parser.add_argument(
                "--eval-metric", default=["NRMS", "SSIM", "EMD", "PSNR"]
            )
        else:
            # 自动检测是否为N28任务
            is_n28_task = (
                "n28" in args.task.lower()
                or (hasattr(args, "ann_file_train") and "N28" in args.ann_file_train)
                or (
                    not hasattr(args, "ann_file_train")
                    and "N14" not in args.task.lower()
                )
            )

            default_dataroot = (
                os.path.join(TRAINING_SET_PATH, "congestion_trainingset1.0")
                if is_n28_task
                else os.path.join(TRAINING_SET_PATH, "congestion")
            )
            default_ann_train = (
                os.path.join(FILES_PATH, "train_N28.csv") 
                if is_n28_task 
                else os.path.join(FILES_PATH, "train_N14.csv")
            )
            default_ann_test = (
                os.path.join(FILES_PATH, "test_N28.csv") 
                if is_n28_task 
                else os.path.join(FILES_PATH, "test_N14.csv")
            )

            print(
                f"Warning: Task '{args.task}' not explicitly defined. Using inferred defaults."
            )
            self.parser.add_argument("--dataroot", default=default_dataroot)
            self.parser.add_argument("--ann_file_train", default=default_ann_train)
            self.parser.add_argument("--ann_file_test", default=default_ann_test)
            self.parser.add_argument("--dataset_type", default="CongestionDataset")
            pass

# 在现有任务配置下添加ISPD2015配置
TASKS.update({
    'congestion_gpdl_ispd2015': {
        'task': 'congestion',
        'ann_file_train': os.path.join(FILES_PATH, 'train_ISPD2015.csv'),
        'ann_file_test': os.path.join(FILES_PATH, 'test_ISPD2015.csv'),
        'dataroot': os.path.join(TRAINING_SET_PATH, 'ISPD2015_congestion'),
        'img_norm_cfg': None,
        'matrix_keys': ['congestion'],
        'model': 'GPDL',
        'model_kwargs': {
            'in_channels': 6,
            'out_channels': 1
        },
        'lr': 1e-4,
        'weight_decay': 5e-4,
        'epoch': 100,
        'steps': [40, 70, 90],
        'batch_size': 4,
        'n_threads': 8,
        'resume_from': None,
        'pretrained': None,
        'save_dir': os.path.join(WORK_DIR, 'congestion_gpdl_ispd2015'),
        'draw_thresh': 0.5,
        'plot_roc': False
    },
    'congestion_gcn_ispd2015': {
        'task': 'congestion',
        'ann_file_train': os.path.join(FILES_PATH, 'train_ISPD2015.csv'),
        'ann_file_test': os.path.join(FILES_PATH, 'test_ISPD2015.csv'),
        'dataroot': os.path.join(TRAINING_SET_PATH, 'ISPD2015_congestion'),
        'img_norm_cfg': None,
        'matrix_keys': ['congestion'],
        'model': 'GCNCongestion',
        'model_kwargs': {
            'in_channels': 6,
            'out_channels': 1,
            'hidden_channels': 64,
            'dropout': 0.5
        },
        'lr': 1e-4,
        'weight_decay': 5e-4,
        'max_iters': 200000,
        'batch_size': 4,
        'n_threads': 8,
        'resume_from': None,
        'pretrained': None,
        'save_dir': os.path.join(WORK_DIR, 'congestion_gcn_ispd2015'),
        'draw_thresh': 0.5,
        'plot_roc': False
    },
    'congestion_visionmamba': {
        'task': 'congestion',
        'ann_file_train': os.path.join(FILES_PATH, 'train_N14.csv'),
        'ann_file_test': os.path.join(FILES_PATH, 'test_N14.csv'),
        'dataroot': os.path.join(TRAINING_SET_PATH, 'congestion'),
        'img_norm_cfg': None,
        'matrix_keys': ['congestion'],
        'model': 'VisionMambaUNet',
        'model_kwargs': {
            'in_channels': 3,
            'out_channels': 1,
            'depths': [2, 2, 2],
            'dims': [32, 64, 128],
            'd_state': 8
        },
        'lr': 2e-4,
        'weight_decay': 0,
        'max_iters': 200000,
        'batch_size': 1,
        'n_threads': 8,
        'resume_from': None,
        'pretrained': None,
        'save_dir': os.path.join(WORK_DIR, 'congestion_visionmamba'),
        'draw_thresh': 0.5,
        'plot_roc': False
    }
})
