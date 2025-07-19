# Copyright 2022 CircuitNet. All rights reserved.
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.losses as losses


def build_loss(opt):
    return losses.__dict__[opt.pop('loss_type')]()


__all__ = ['L1Loss', 'MSELoss', 'BCEWithLogitsLoss', 'VAELoss']  # 添加VAELoss


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()
    return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction='mean', sample_wise=False):
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        eps = 1e-12
        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        else:
            loss = loss.sum() / (weight.sum() + eps)
    return loss


def masked_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', sample_wise=False, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss

    return wrapper


@masked_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@masked_loss
def bce_with_logits_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # 新增基础函数


# VAE重建损失函数
@masked_loss
def vae_recon_loss(pred, target, loss_type='mse'):
    if loss_type == 'mse':
        return F.mse_loss(pred, target, reduction='none')
    elif loss_type == 'l1':
        return F.l1_loss(pred, target, reduction='none')
    else:
        raise ValueError(f"Unknown reconstruction loss type: {loss_type}")


class L1Loss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction, sample_wise=self.sample_wise)


class MSELoss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction, sample_wise=self.sample_wise)


class BCEWithLogitsLoss(nn.Module):  # 新增类
    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * bce_with_logits_loss(pred, target, weight, reduction=self.reduction,
                                                       sample_wise=self.sample_wise)


class VAELoss(nn.Module):
    """
    变分自编码器损失函数，结合重建损失和KL散度

    Args:
        recon_weight (float): 重建损失的权重，默认100.0
        kl_weight (float): KL散度的权重，默认1.0
        reduction (str): 降维方式，默认'mean'
        sample_wise (bool): 是否按样本计算，默认False
        recon_type (str): 重建损失类型，默认'mse'，可选'l1'
    """

    def __init__(self, recon_weight=100.0, kl_weight=1.0, reduction='mean',
                 sample_wise=False, recon_type='mse'):
        super(VAELoss, self).__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.recon_type = recon_type

    def forward(self, pred_tuple, target, weight=None, **kwargs):
        """
        计算VAE损失

        Args:
            pred_tuple (tuple): 包含(重建输出, 均值, 对数方差)的元组
            target (Tensor): 目标数据
            weight (Tensor, optional): 掩码权重

        Returns:
            Tensor: 总损失值
        """
        # 解包预测结果
        if isinstance(pred_tuple, tuple) and len(pred_tuple) == 3:
            pred, mu, logvar = pred_tuple
        else:
            raise ValueError("VAELoss expects pred_tuple to be (output, mu, logvar)")

        # 重建损失
        recon_loss = self.recon_weight * vae_recon_loss(
            pred, target, weight=weight, reduction=self.reduction,
            sample_wise=self.sample_wise, loss_type=self.recon_type
        )

        # KL散度: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # 注意：不使用掩码，因为KL散度应用于整个分布
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 如果是batch维度的均值
        if self.reduction == 'mean':
            batch_size = pred.size(0)
            kl_div = kl_div / batch_size

        # 总损失
        loss = recon_loss + self.kl_weight * kl_div

        return loss

    def update_kl_weight(self, new_weight):
        """动态更新KL权重"""
        self.kl_weight = new_weight
