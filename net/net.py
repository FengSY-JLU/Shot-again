import torch
from net.ITA import JNet, TNet

class net(torch.nn.Module):
    def __init__(self, use_spb: bool = True, use_sgca: bool = True):
        super().__init__()
        # 将开关透传到两个分支
        self.image_net = JNet(use_spb=use_spb, use_sgca=use_sgca)
        self.mask_net  = TNet(use_spb=use_spb, use_sgca=use_sgca)

    def forward(self, data):
        x_j_all = self.image_net(data)
        x_t_all = self.mask_net(data)
        x_j = x_j_all[-1]
        x_t = x_t_all[-1]
        return x_j, x_t
