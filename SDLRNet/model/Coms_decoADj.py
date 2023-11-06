from pip import main
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR + "-------------------------------")
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from model.conv_trans_utils import *




class CoMs_deAdj(nn.Module):
    #  num_croblock=[0,0,0,0]
    def __init__(self, in_chan, base_chan, num_classes=9, num_block=0, num_croblock=[0, 0, 2, 0], trans=False,
                 num_heads=[3, 6, 12], attn_drop=0., bottleneck=False, maxpool=True, mode="train",
                 n_groups=[1, 1, 3, 6]):
        super().__init__()

        self.inc = [BasicBlock(in_chan, base_chan), BasicBlock(base_chan, base_chan)]
        self.mode = mode
        # 卷积提取特征
        for i in range(num_block):
            self.inc.append(BasicBlock(base_chan, base_chan))

        self.inc = nn.Sequential(*self.inc)

        # 下采样
        # num_block决定卷积block数
        self.down1 = down_block_transBD(base_chan, 2 * base_chan, bottleneck=bottleneck, trans=trans,
                                        maxpool=maxpool, heads=num_heads[0], num_croblock=num_croblock[0],
                                        attn_drop=attn_drop, n_groups=n_groups[0], imgsize=112)
        self.down2 = down_block_transBD(2 * base_chan, 96, bottleneck=bottleneck, num_croblock=num_croblock[1],
                                        maxpool=maxpool, heads=num_heads[0], attn_drop=attn_drop, n_groups=n_groups[1],
                                        imgsize=56)
        self.down3 = down_block_transBD(96, 192, bottleneck=bottleneck, num_croblock=num_croblock[2],
                                        maxpool=maxpool, heads=num_heads[1], attn_drop=attn_drop, n_groups=n_groups[1],
                                        imgsize=28)
        self.down4 = down_block_transBD(192, 384, bottleneck=bottleneck, num_croblock=num_croblock[3],
                                        maxpool=maxpool, heads=num_heads[2], attn_drop=attn_drop, n_groups=n_groups[2],
                                        imgsize=14)

        ####调整通道
        self.Adch4 = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.Adch3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.Adch2 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.Adch1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(64), nn.PReLU(),

        )
        self.fuse1 = Chan_spaAtt(256, 64, 4)
        # in_ch,map_ch,out_ch
        # self.attention4 = BidirectionAttention(384, 64, 384, heads=num_heads[2], dim_head=32, attn_drop=attn_drop,
        #                                        proj_drop=attn_drop)
        # self.attention3 = BidirectionAttention(192, 64, 192, heads=num_heads[1], dim_head=32, attn_drop=attn_drop,
        #                                        proj_drop=attn_drop)
        # self.attention2 = BidirectionAttention(96, 64, 96, heads=num_heads[0], dim_head=32, attn_drop=attn_drop,
        #                                        proj_drop=attn_drop)
        # self.attention1 = BidirectionAttention(64, 64, 64, heads=2, dim_head=32, attn_drop=attn_drop,
        #                                        proj_drop=attn_drop)

        self.up1 = up_block_transADj(384, 192, bottleneck=bottleneck,
                                  heads=num_heads[1], attn_drop=attn_drop)
        self.up2 = up_block_transADj(192, 96, bottleneck=bottleneck,
                                  heads=num_heads[0], attn_drop=attn_drop)
        self.up3 = up_block_transADj(96, 2 * base_chan, bottleneck=bottleneck,
                                  heads=2, attn_drop=attn_drop)
        self.up4 = up_block_transADj(2 * base_chan, base_chan, bottleneck=bottleneck,
                                  heads=1, attn_drop=attn_drop)

        self.mlp = Mlp(base_chan, base_chan)
        # 最后输出 直接1x1卷积 把维度降为类别数 就是分割结果
        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        outlist = []
        x = self.inc(x)
        # outlist.append(x)
        # x1 = x
        x1 = self.down1(x)

        x2 = self.down2(x1)
        outlist.append(x2)
        x3 = self.down3(x2)
        outlist.append(x3)
        x4 = self.down4(x3)
        outlist.append(x4)

        # 上采样到56，56
        down4 = F.interpolate(self.Adch4(x4), size=x2.size()[2:], mode='bilinear')
        down3 = F.interpolate(self.Adch3(x3), size=x2.size()[2:], mode='bilinear')
        down2 = F.interpolate(self.Adch2(x2), size=x2.size()[2:], mode='bilinear')
        down1 = self.Adch1(x1)
        # 多尺度融合
        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))
        print("----fus---shape---",fuse1.shape)
        # # 多尺度与单尺度
        # f4 = self.attention4(x4, fuse1)
        # f3 = self.attention3(x3, fuse1)
        # f2 = self.attention2(x2, fuse1)
        # f1 = self.attention1(x1, fuse1)

        out1 = self.up1(x4, x3,fuse1)
        out2 = self.up2(out1, x2,fuse1)
        out3 = self.up3(out2, x1,fuse1)
        out4 = self.up4(out3, x,None)
        # if self.mode == "train":
        out = self.mlp(out4)
        # outlist.append(out)
        out = self.outc(out)
        # outlist.append(out)

        return out


if __name__ == '__main__':
    net = CoMs_deAdj(3, 32, 9, num_heads=[3, 6, 12, 24], num_block=0, num_croblock=[0, 0, 0, 0],
                        attn_drop=0.1, maxpool=True)
    print(net)
    # net.to("cuda")

    # from torchsummary import summary
    # summary(net,(3,224,224))
    with torch.autograd.detect_anomaly():
        input = torch.rand(3, 1, 224, 224)
        # out [3,4,256,256]
        # 1 torch.Size([3, 32, 256, 256])
        # torch.Size([3, 64, 128, 128]) torch.Size([3, 128, 64, 64])
        # torch.Size([3, 256, 32, 32]) torch.Size([3, 512, 16, 16])
        outp = net(input)
        outp.sum().backward()

    print(outp.size())

