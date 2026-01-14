import torch
import torch.nn as nn
from net import SGCA

class StructPreserveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StructPreserveBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 定义3x1和1x3的卷积层
        self.conv3x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))

        # 定义FeatureGate层
        self.feature_gate = FeatureGate(out_channels)

        # 定义第二个3x1和1x3的卷积层
        self.conv3x1_2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x3_2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=(1, 3), padding=(0, 1))

        # 定义1x1的卷积层用于输出
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # self.sge = SGCA.SpatialGroupCrosschannelAttention(groups=4)

        # 定义1x1的卷积层用于残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 保存输入张量，用于残差连接
        residual = self.residual_conv(x)

        # 分别对输入进行3x1和1x3的卷积操作
        out_3x1_1 = self.conv3x1_1(x)
        out_1x3_1 = self.conv1x3_1(x)

        # 将卷积结果进行加运算
        combined_features = out_3x1_1 + out_1x3_1

        # 将合成后的结果输入到FeatureGate中
        gated_features = self.feature_gate(combined_features)

        # 再次对输出进行3x1和1x3的卷积操作
        out_3x1_2 = self.conv3x1_2(gated_features)
        out_1x3_2 = self.conv1x3_2(gated_features)

        # 将再次卷积的结果进行加运算
        final_output = out_3x1_2 + out_1x3_2

        # final_output = self.sge(final_output)

        # 通过1x1卷积进行输出
        final_output = self.output_conv(final_output)

        # 添加残差连接
        final_output += residual

        return final_output

class FeatureGate(nn.Module):
    def __init__(self, in_channels):
        super(FeatureGate, self).__init__()

        self.in_channels = in_channels

    def forward(self, x):
        # 对输入张量按通道进行分割
        split = torch.split(x, self.in_channels // 2, dim=1)

        # 分别对两部分进行逐通道的元素乘积
        gated_features = split[0] * split[1]

        return gated_features

if __name__ == '__main__':

    # 测试代码
    # 定义一个输入张量大小为 (batch_size, channels, height, width)
    batch_size = 1
    channels = 64
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)

    # 创建模型实例
    model = StructPreserveBlock(in_channels=channels, out_channels=channels)

    # 执行前向传播
    output = model(input_tensor)

    # 打印输出张量大小
    print("输出张量大小:", output.size())