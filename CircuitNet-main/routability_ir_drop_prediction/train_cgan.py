# Copyright 2022 CircuitNet. All rights reserved.

import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
from models.build_model import build_model
from utils.configs import Parser
from math import cos, pi
import sys


def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"{save_path}/model_iters_{epoch}.pth"
    torch.save({"state_dict": model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def plot_losses(losses, save_path, title="训练损失曲线"):
    """绘制并保存损失曲线图"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss")
    plt.title(title)
    plt.xlabel("迭代次数 (x100)")
    plt.ylabel("损失值")
    plt.grid(True)
    plt.legend()

    # 添加当前最低损失值的标注
    if losses:
        min_loss = min(losses)
        min_idx = losses.index(min_loss)
        plt.annotate(
            f"最低损失: {min_loss:.4f}",
            xy=(min_idx, min_loss),
            xytext=(min_idx, min_loss * 1.3),
            arrowprops=dict(facecolor="red", shrink=0.05),
        )

    # 保存图像
    os.makedirs(os.path.join(save_path, "visualization"), exist_ok=True)
    plt.savefig(os.path.join(save_path, "visualization", "loss_curve.png"))
    plt.close()


def plot_gan_losses(d_losses, g_losses, g_adv_losses, g_l1_losses, save_path):
    """绘制CGAN多损失曲线图"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(d_losses, label="判别器损失")
    plt.title("判别器损失")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(g_losses, label="生成器总损失")
    plt.title("生成器总损失")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(g_adv_losses, label="生成器对抗损失")
    plt.title("生成器对抗损失")
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(g_l1_losses, label="生成器L1损失")
    plt.title("生成器L1损失")
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs(os.path.join(save_path, "visualization"), exist_ok=True)
    plt.savefig(os.path.join(save_path, "visualization", "gan_losses.png"))
    plt.close()


class CosineRestartLr(object):
    def __init__(
        self, base_lr, periods, restart_weights=[1], min_lr=None, min_lr_ratio=None
    ):
        # 确保periods中的元素都是整数
        self.periods = [int(p) for p in periods]
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(
        self, start: float, end: float, factor: float, weight: float = 1.0
    ) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(
            f"Current iteration {iteration} exceeds "
            f"cumulative_periods {cumulative_periods}"
        )

    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group["lr"] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault("initial_lr", group["lr"])
            self.base_lr = [
                group["initial_lr"] for group in optimizer.param_groups  # type: ignore
            ]


def train_cgan():
    """
    专门用于训练CGAN模型的函数
    """
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, "rt") as f:
            arg_dict.update(json.load(f))
    
    # 确保某些参数是整数类型
    arg_dict["max_iters"] = int(arg_dict["max_iters"])
    
    # 确保是CGAN模型
    if arg_dict["model_type"] != "CGANCongestion":
        print("错误：此训练脚本仅用于CGAN模型，请使用--task congestion_cgan")
        return
    
    if not os.path.exists(arg_dict["save_path"]):
        os.makedirs(arg_dict["save_path"])
    with open(os.path.join(arg_dict["save_path"], "arg.json"), "wt") as f:
        json.dump(arg_dict, f, indent=4)

    # 创建可视化目录
    vis_dir = os.path.join(arg_dict["save_path"], "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    arg_dict["ann_file"] = arg_dict["ann_file_train"]
    arg_dict["test_mode"] = False

    print("===> Loading datasets")
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    print("===> Building model")
    # Initialize model parameters
    model = build_model(arg_dict)
    if not arg_dict["cpu"]:
        model = model.cuda()

    # 为生成器和判别器分别创建优化器
    g_optimizer = optim.AdamW(
        model.generator.parameters(),
        lr=arg_dict["lr"],
        betas=(0.5, 0.999),  # GAN训练推荐的beta值
        weight_decay=arg_dict["weight_decay"],
    )
    
    d_optimizer = optim.AdamW(
        model.discriminator.parameters(),
        lr=arg_dict["lr"],
        betas=(0.5, 0.999),  # GAN训练推荐的beta值
        weight_decay=arg_dict["weight_decay"],
    )

    # Build lr scheduler
    g_cosine_lr = CosineRestartLr(arg_dict["lr"], [arg_dict["max_iters"]], [1], 1e-7)
    d_cosine_lr = CosineRestartLr(arg_dict["lr"], [arg_dict["max_iters"]], [1], 1e-7)
    
    g_cosine_lr.set_init_lr(g_optimizer)
    d_cosine_lr.set_init_lr(d_optimizer)

    # 用于记录损失
    all_losses = []  # 总损失
    d_losses = []    # 判别器损失历史
    g_losses = []    # 生成器损失历史
    g_adv_losses = [] # 生成器对抗损失历史
    g_l1_losses = [] # 生成器L1损失历史
    
    # 当前训练周期的损失记录
    curr_d_losses = []
    curr_g_losses = []
    curr_g_adv_losses = []
    curr_g_l1_losses = []

    # 如果存在之前的损失记录，则加载
    loss_file = os.path.join(vis_dir, "losses.npy")
    if os.path.exists(loss_file):
        try:
            all_losses = np.load(loss_file).tolist()
            print(f"加载了 {len(all_losses)} 个之前的损失记录")
        except Exception as e:
            print(f"加载损失记录失败: {e}")

    # GAN超参数
    lambda_L1 = 1.0  # L1损失权重 (从100.0降低到1.0避免震荡)
    
    # 早停机制参数
    best_loss = float('inf')
    patience = 15  # GAN训练用较短的耐心值
    patience_counter = 0
    
    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 1000
    vis_freq = 500  # 可视化频率
    
    print(f"开始CGAN训练，L1权重: {lambda_L1}，使用梯度裁剪和早停机制...")

    while iter_num < arg_dict["max_iters"]:
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in dataset:
                if arg_dict["cpu"]:
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

                # 更新判别器和生成器的学习率
                g_regular_lr = g_cosine_lr.get_regular_lr(iter_num)
                d_regular_lr = d_cosine_lr.get_regular_lr(iter_num)
                g_cosine_lr._set_lr(g_optimizer, g_regular_lr)
                d_cosine_lr._set_lr(d_optimizer, d_regular_lr)

                # 生成随机噪声
                batch_size = input.size(0)
                noise_dim = arg_dict["noise_dim"]
                noise = torch.randn(batch_size, noise_dim)
                if not arg_dict["cpu"]:
                    noise = noise.cuda()
                
                # -----------------
                # 训练判别器
                # -----------------
                d_optimizer.zero_grad()
                
                # 生成假样本
                fake_images = model(input, noise)
                
                # 真实样本的判别结果
                real_validity = model.discriminate(input, target)
                # 假样本的判别结果
                fake_validity = model.discriminate(input, fake_images.detach())
                
                # 判别器损失
                d_real_loss = torch.nn.functional.binary_cross_entropy(
                    real_validity, 
                    torch.ones_like(real_validity)
                )
                d_fake_loss = torch.nn.functional.binary_cross_entropy(
                    fake_validity, 
                    torch.zeros_like(fake_validity)
                )
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                # 更新判别器 - 添加梯度裁剪
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                d_optimizer.step()
                
                # -----------------
                # 训练生成器
                # -----------------
                g_optimizer.zero_grad()
                
                # 生成假样本
                fake_images = model(input, noise)
                # 假样本的判别结果
                fake_validity = model.discriminate(input, fake_images)
                
                # 生成器对抗损失
                g_adv_loss = torch.nn.functional.binary_cross_entropy(
                    fake_validity, 
                    torch.ones_like(fake_validity)
                )
                
                # 生成器L1损失 (帮助生成更清晰的图像)
                g_l1_loss = torch.nn.functional.l1_loss(fake_images, target)
                
                # 总生成器损失
                g_loss = g_adv_loss + lambda_L1 * g_l1_loss
                
                # 更新生成器 - 添加梯度裁剪
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
                g_optimizer.step()
                
                # 记录损失
                curr_d_losses.append(d_loss.item())
                curr_g_losses.append(g_loss.item())
                curr_g_adv_losses.append(g_adv_loss.item())
                curr_g_l1_losses.append(g_l1_loss.item())
                
                # 总损失 (用于兼容原有的可视化)
                total_loss = d_loss + g_loss
                epoch_loss += total_loss.item()
                
                iter_num += 1
                
                bar.update(1)
                
                if iter_num % print_freq == 0:
                    break

        # 计算平均损失并记录
        avg_loss = epoch_loss / print_freq
        all_losses.append(avg_loss)
        
        # 计算和记录当前周期的各项平均损失
        avg_d_loss = sum(curr_d_losses) / len(curr_d_losses) if curr_d_losses else 0
        avg_g_loss = sum(curr_g_losses) / len(curr_g_losses) if curr_g_losses else 0
        avg_g_adv_loss = sum(curr_g_adv_losses) / len(curr_g_adv_losses) if curr_g_adv_losses else 0
        avg_g_l1_loss = sum(curr_g_l1_losses) / len(curr_g_l1_losses) if curr_g_l1_losses else 0
        
        # 添加到历史记录
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        g_adv_losses.append(avg_g_adv_loss)
        g_l1_losses.append(avg_g_l1_loss)
        
        # 早停机制检查 (基于生成器损失)
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_path = os.path.join(arg_dict["save_path"], "best_model.pth")
            torch.save({"state_dict": model.state_dict(), "loss": best_loss}, best_model_path)
        else:
            patience_counter += 1
        
        # 清空当前周期的损失记录
        curr_d_losses = []
        curr_g_losses = []
        curr_g_adv_losses = []
        curr_g_l1_losses = []
        
        # 获取当前学习率
        current_lr = g_optimizer.param_groups[0]['lr']

        print(
            "===> Iters[{}]({}/{}): D_Loss: {:.4f}, G_Loss: {:.4f}, G_Adv: {:.4f}, G_L1: {:.4f}, Best: {:.4f}, LR: {:.2e}, Patience: {}/{}".format(
                iter_num, iter_num, arg_dict["max_iters"], 
                avg_d_loss, avg_g_loss, avg_g_adv_loss, avg_g_l1_loss, best_loss,
                current_lr, patience_counter, patience
            )
        )
        
        # 早停检查
        if patience_counter >= patience:
            print(f"早停触发！连续{patience}个周期没有改善，停止训练")
            break

        # 定期可视化损失曲线
        if iter_num % vis_freq == 0:
            # 绘制总损失曲线 (兼容原有可视化)
            plot_losses(
                all_losses,
                arg_dict["save_path"],
                f"训练损失曲线 (迭代次数: {iter_num})",
            )
            
            # 绘制GAN专用的多损失曲线
            plot_gan_losses(
                d_losses, g_losses, g_adv_losses, g_l1_losses,
                arg_dict["save_path"]
            )

            # 保存损失数据
            np.save(os.path.join(vis_dir, "losses.npy"), np.array(all_losses))
            np.save(os.path.join(vis_dir, "d_losses.npy"), np.array(d_losses))
            np.save(os.path.join(vis_dir, "g_losses.npy"), np.array(g_losses))
            np.save(os.path.join(vis_dir, "g_adv_losses.npy"), np.array(g_adv_losses))
            np.save(os.path.join(vis_dir, "g_l1_losses.npy"), np.array(g_l1_losses))
            
            print(f"损失曲线已保存至: {vis_dir}/loss_curve.png 和 {vis_dir}/gan_losses.png")

        # 定期保存模型
        if iter_num % save_freq == 0:
            checkpoint(model, iter_num, arg_dict["save_path"])

        epoch_loss = 0

    # 训练结束，绘制最终损失曲线
    plot_losses(all_losses, arg_dict["save_path"], "训练完整损失曲线")
    plot_gan_losses(
        d_losses, g_losses, g_adv_losses, g_l1_losses,
        arg_dict["save_path"]
    )
    print(f"训练完成！最终损失曲线已保存至: {vis_dir}/loss_curve.png 和 {vis_dir}/gan_losses.png")
    print(f"最佳生成器损失: {best_loss:.4f}，已保存至: {arg_dict['save_path']}/best_model.pth")


if __name__ == "__main__":
    train_cgan() 