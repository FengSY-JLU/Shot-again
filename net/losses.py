import torch
from torch import nn
import torch.nn.functional as F


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = torch.pow(mr-0.5, 2)
        Dg = torch.pow(mg-0.5, 2)
        Db = torch.pow(mb-0.5, 2)
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Dg, 2) + torch.pow(Db, 2), 0.5)
        return k

class ColorLossWithRegularization(nn.Module):
    def __init__(self, regularization_weight):
        super(ColorLossWithRegularization, self).__init__()
        self.regularization_weight = regularization_weight

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = torch.pow(mr - 0.5, 2)
        Dg = torch.pow(mg - 0.5, 2)
        Db = torch.pow(mb - 0.5, 2)
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Dg, 2) + torch.pow(Db, 2), 0.5)

        # 计算红色通道的偏差
        red_deviation = torch.abs(mr - 0.5)

        # 添加正则化项，惩罚红色通道的偏差
        regularization_term = torch.mean(torch.pow(red_deviation, 2))

        # 将正则化项与原损失函数相加
        loss = k + self.regularization_weight * regularization_term

        return loss

class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, original_img, compressed_img):
        # 将图像转换为浮点数类型
        original_img = original_img.float()
        compressed_img = compressed_img.float()

        # 计算 MSE (Mean Squared Error)
        mse = F.mse_loss(original_img, compressed_img)

        # 如果 MSE 非零，计算 PSNR
        if mse == 0:
            return float('inf')
        else:
            max_pixel = 255.0  # 输入图像范围是 [0, 1]
            psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
            return 1.0 / psnr  # 返回 PSNR的倒数，因为我们希望最小化损失