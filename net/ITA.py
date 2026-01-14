import torch
from net import SGCA
from net import SPB

class JNet(torch.nn.Module):
    def __init__(self, num=64, use_spb: bool = True, use_sgca: bool = True):
        super().__init__()
        self.use_spb = use_spb
        self.use_sgca = use_sgca

        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )

        # 模块保持原实现，仅新增“插拔”能力
        self.spb4 = SPB.StructPreserveBlock(in_channels=num, out_channels=num)
        self.sge4 = SGCA.SpatialGroupCrosschannelAttention(groups=4)

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(num, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        out1 = self.conv1(data)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)

        # —— SPB 插入点：conv4 之后
        if self.use_spb:
            out4 = self.spb4(out4)

        # —— SGCA 插入点：SPB 之后（或直接用 conv4 输出）
        if self.use_sgca:
            out5 = self.sge4(out4)
        else:
            out5 = out4

        final_out = self.final(out5)
        return out1, out2, out3, out4, out5, final_out


class TNet(torch.nn.Module):
    def __init__(self, num=64, use_spb: bool = True, use_sgca: bool = True):
        super().__init__()
        self.use_spb = use_spb
        self.use_sgca = use_sgca

        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )

        self.spb4 = SPB.StructPreserveBlock(in_channels=num, out_channels=num)
        self.sge4 = SGCA.SpatialGroupCrosschannelAttention(groups=4)

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(num, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        out1 = self.conv1(data)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)

        if self.use_spb:
            out4 = self.spb4(out4)
        if self.use_sgca:
            out5 = self.sge4(out4)
        else:
            out5 = out4

        final_out = self.final(out5)
        return out1, out2, out3, out4, out5, final_out
