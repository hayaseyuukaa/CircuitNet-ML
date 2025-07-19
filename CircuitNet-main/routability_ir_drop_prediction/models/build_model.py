import models
import torch


def build_model(opt):
    opt = dict(opt)
    model_type = opt.pop("model_type")

    # 定义各模型支持的参数
    valid_keys = {
        "GATCongestion": [
            "in_channels",
            "out_channels",
            "hidden_channels",
            "heads",
            "dropout",
        ],
        "GPDL": ["in_channels", "out_channels"],
        "RouteNet": ["in_channels", "out_channels"],
        "MAVI": ["in_channels", "out_channels"],
        # "HybridModel": ["in_channels", "out_channels"],  # 注释掉不存在的模型
        "GCNCongestion": ["in_channels", "out_channels", "hidden_channels", "dropout"],
        "CGANCongestion": ["in_channels", "out_channels", "noise_dim"],
        "CVAECongestion": [
            "in_channels",
            "out_channels",
            "latent_dim",
            "kld_weight",
            "use_mmd",
            "mmd_weight",
            "use_feature_matching",
            "feature_matching_weight",
        ],
        "FCNCongestion": ["in_channels", "out_channels"],
        "LHNNCongestion": ["in_channels", "out_channels"],
        "GraphSAGECongestion": ["in_channels", "out_channels", "hidden_channels", "dropout"],
        "RGCNCongestion": ["in_channels", "out_channels", "hidden_channels", "dropout"],
        "SwinTransformerCongestion": [
            "in_channels",
            "out_channels",
            "embed_dim",
            "depths",
            "num_heads",
            "window_size",
            "mlp_ratio",
            "drop_rate",
            "attn_drop_rate",
            "drop_path_rate",
            "ape",
            "patch_norm",
            "use_checkpoint",
            "img_size",
        ],
        "Congestion_Prediction_Net": ["in_channels", "out_channels"],
        "DRC_Prediction_Net": ["in_channels", "out_channels"],
        # 添加BRAU-Net++模型的参数支持
        "BRAUNet_Congestion": ["in_channels", "out_channels", "img_size", "n_win"],
    }

    # 参数过滤逻辑
    if model_type in valid_keys:
        model_opt = {k: opt[k] for k in valid_keys[model_type] if k in opt}
    else:
        model_opt = opt

    # 模型名称映射
    model_display_names = {
        "Congestion_Prediction_Net": "ibUNet (Inception Boosted U-Net)",
        "GPDL": "GPDL",
        "GCNCongestion": "GCN",
        "EGEUNetCongestion": "EGE-UNet",
        "GraphSAGECongestion": "GraphSAGE",
        "RGCNCongestion": "RGCN",
        "RouteNet": "RouteNet",
        "SwinTransformerCongestion": "Swin Transformer",
        "VMUNet": "VM-UNet",
        "BRAUNet_Congestion": "BRAU-Net++",
        "GATCongestion": "GAT"
    }

    display_name = model_display_names.get(model_type, model_type)

    # 实例化模型
    try:
        model = models.__dict__[model_type](**model_opt)
        print(f"成功创建模型: {display_name}")
    except Exception as e:
        print(f"创建模型 {display_name} 失败: {e}")
        raise

    # 初始化权重 - 只传递init_weights所需的参数
    if hasattr(model, "init_weights"):
        # 提取init_weights方法所需的参数
        init_weights_params = {
            "pretrained": opt.get("pretrained"),
            "strict": opt.get("strict", True)
        }
        
        if "load_state_dict" in opt:
            init_weights_params["load_state_dict"] = opt.get("load_state_dict")
            
        model.init_weights(**init_weights_params)

    return model
