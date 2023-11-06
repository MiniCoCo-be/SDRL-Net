from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import pdb
import einops
import numpy as np
from einops.layers.torch import Rearrange
from torch import nn, einsum
from timm.models.layers import to_2tuple, trunc_normal_

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

# 深度可分离生成q,k,v
class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x): 
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out 

class Mlp(nn.Module):
    def __init__(self, in_ch, hid_ch=None, out_ch=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_ch = out_ch or in_ch
        hid_ch = in_ch * 4

        self.fc1 = nn.Conv2d(in_ch, hid_ch, kernel_size=1)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(hid_ch)
        self.fc2 = nn.Conv2d(hid_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x



# 更宽的网络50层、101层的基本块 通道更大
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64, sd=0.0,
                 **block_kwargs):
        super(Bottleneck, self).__init__()

        width = int(channels * (width_per_group / 64.)) * groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(conv1x1(in_channels, channels * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        self.conv1 = conv1x1(in_channels, width)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(width),
            nn.ReLU(),
            conv3x3(width, width, stride=stride, groups=groups),
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(width),
            nn.ReLU(),
            conv1x1(width, channels * self.expansion),
        )

        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.bn(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.bn(x)
            x = self.relu(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.sd(x) + skip

        return x


# 残差卷积提取特征
class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu, 
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x): 
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out

# transEncoder提取特征
class BasicTransBlock(nn.Module):

    def __init__(self, in_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)

        self.attn = LinearAttention(in_ch, heads=heads, dim_head=in_ch//heads, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.gelu = nn.GeLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        # conv1x1 has not difference with mlp in performance

    def forward(self, x):

        out = self.bn1(x)
        # out = self.gelu(out)

        out, q_k_attn = self.attn(out)
        
        out = out + x
        residue = out

        out = self.bn2(out)
        out = self.gelu(out)
        out = self.mlp(out)

        out += residue

        return out

# 低特征上采样 两个特征进行融合 低级特征与高级特征自注意再融合
class BasicTransDecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()

        self.bn_l = nn.BatchNorm2d(in_ch)
        self.bn_h = nn.BatchNorm2d(out_ch)

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.attn = LinearAttentionDecoder(in_ch, out_ch, heads=heads, dim_head=out_ch//heads, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        # 低特征上采样
        residue = F.interpolate(self.conv_ch(x1), size=x2.shape[-2:], mode='bilinear', align_corners=True)
        #x1: low-res, x2: high-res
        x1 = self.bn_l(x1)
        x2 = self.bn_h(x2)
        # 主要使用低分辨率特征
        out, q_k_attn = self.attn(x2, x1)
        
        out = out + residue
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DAttentionBaseline(nn.Module):
    #  offset_range_factor[1,2,3,4] n_groups=-1,-1,3,6  n_heads=3,6,12,24
    def __init__(
            self, q_size=56, kv_size=56, n_heads=8, n_head_channels=32, n_groups=1,
            attn_drop=0.1, proj_drop=0.1, stride=1,
            offset_range_factor=1, use_pe=False, dwc_pe=False,
            no_off=False, fixed_pe=False
    ):

        super().__init__()
        q_size = to_2tuple(q_size)
        kv_size = to_2tuple(kv_size)
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        if self.q_h == 14 or self.q_w == 14 or self.q_h == 24 or self.q_w == 24:
            kk = 5
        elif self.q_h == 7 or self.q_w == 7 or self.q_h == 12 or self.q_w == 12:
            kk = 3
        elif self.q_h == 28 or self.q_w == 28 or self.q_h == 48 or self.q_w == 48:
            kk = 7
        elif self.q_h == 56 or self.q_w == 56 or self.q_h == 96 or self.q_w == 96:
            kk = 9
        # elif self.q_h == 112 or self.q_w == 112 or self.q_h == 192 or self.q_w == 192:
        #     kk = 11

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc,
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe:

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_ref_points(H, W, B, dtype, device)

                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)

                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)

                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class AttentionBlockB_D(nn.Module):
    # Attention block with pre-activation.
    expansion = 1

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, sd=0.0,stride=1, window_size=7, k=1,
                  norm=nn.BatchNorm2d, activation=nn.GELU, imgsize, n_groups=1, offset_range_factor=1,
                 **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        # attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion

        self.shortcut = []
        if stride != 1 or dim_in != dim_out * self.expansion:
            self.shortcut.append(conv1x1(dim_in, dim_out * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.norm1 = norm(dim_in)
        self.relu = activation()

        self.conv = nn.Conv2d(dim_in, width, kernel_size=1, bias=False)
        self.norm2 = norm(width)
        self.attns = []
        # self.attns.append(attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout))
        if imgsize != 112:
            dim_head = dim_out // heads
            print("heads:{},dim_out:{}".format(heads, dim_out))
            assert dim_out == heads * dim_head, "dim_C与dim_head不匹配"
            self.attns.append(
                DAttentionBaseline(q_size=imgsize, kv_size=imgsize, n_heads=heads, n_head_channels=dim_head,
                                   n_groups=n_groups, offset_range_factor=offset_range_factor))
        self.attns = nn.Sequential(*self.attns)

        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.norm1(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.conv(x)
        x = self.norm2(x)
        x, pos, ref = self.attns(x)

        x = self.sd(x) + skip

        return x


###############################conv--DA###################################3
class down_block_transBD(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=False, maxpool=True, num_croblock=0, trans=False,
                 heads=4, attn_drop=0.0, n_groups=1, imgsize=56,offset_range_factor=2):

        super().__init__()

        block_list = []

        if bottleneck:
            block = None
            # block = BottleneckBlock
        else:
            block = BasicBlock

        attn_block = AttentionBlockB_D
        # gl_attblock = BasicTransBlock

        if maxpool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch, stride=1))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        if imgsize != 112:
            block_list.append(attn_block(out_ch, out_ch, heads=heads, sd=attn_drop, n_groups=n_groups, imgsize=imgsize,offset_range_factor=offset_range_factor))

        for i in range(num_croblock):
            block_list.append(block(out_ch, out_ch, stride=1))
            block_list.append(attn_block(out_ch, out_ch, heads=heads, sd=attn_drop, n_groups=n_groups, imgsize=imgsize))

        self.blocks = nn.Sequential(*block_list)

    def forward(self, x):

        out = self.blocks(x)
        return out

class upAttention(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads=4, dim_head=64, attn_drop=0.,
                 proj_drop=0., proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.GELU):
        super().__init__()

        # 更改map_dim维度
        self.map_projection = nn.Conv2d(map_dim, out_dim, kernel_size=1, bias=False)
        self.map_dim = out_dim

        self.inner_dim = dim_head * heads
        self.feat_dim = feat_dim

        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head


        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.norm1 = norm(feat_dim) if norm else nn.Identity()  # norm layer for feature map
        self.norm2 = norm(self.map_dim) if norm else nn.Identity()  # norm layer for semantic map

        assert proj_type in ['linear', 'depthwise']

        if proj_type == 'linear':
            self.feat_qv = nn.Conv2d(feat_dim, self.inner_dim * 2, kernel_size=1, bias=False)
            self.feat_out = nn.Conv2d(self.inner_dim, out_dim, kernel_size=1, bias=False)

        else:
            self.feat_qv = depthwise_separable_conv(feat_dim, self.inner_dim * 2)
            self.feat_out = depthwise_separable_conv(self.inner_dim, out_dim)

        self.map_qv = nn.Conv2d(self.map_dim, self.inner_dim * 2, kernel_size=1, bias=False)
        # self.map_out = nn.Conv2d(self.inner_dim, map_dim, kernel_size=1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat, semantic_map):

        B, C, H, W = feat.shape
        B, C, Hh, Wh = feat.shape

        res = feat
        feat = self.norm1(feat)
        semantic_map = self.map_projection(semantic_map)
        semantic_map = self.norm2(semantic_map)

        feat_q, feat_v = self.feat_qv(feat).chunk(2, dim=1)  # B, inner_dim, H, W
        map_q, map_v = self.map_qv(semantic_map).chunk(2, dim=1)  # B, inner_dim, rs, rs

        feat_q, feat_v = map(
            lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                heads=self.heads, h=H, w=W), [feat_q, feat_v])
        map_q, map_v = map(
            lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                heads=self.heads, h=Hh, w=Wh), [map_q, map_v])
            # lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
            #                     heads=self.heads, h=self.map_size, w=self.map_size), [map_q, map_v])
        #交叉注意 双q
        attn = torch.einsum('bhid,bhjd->bhij', feat_q, map_q)
        attn *= self.scale
        # 对map语义那维进行softmax
        feat_map_attn = F.softmax(attn, dim=-1)  # semantic map is very concise that don't need dropout
        # add dropout migth cause unstable during training
        # map_feat_attn = self.attn_drop(F.softmax(attn, dim=-2))

        feat_out = torch.einsum('bhij,bhjd->bhid', feat_map_attn, map_v)
        feat_out = rearrange(feat_out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W,
                             dim_head=self.dim_head, heads=self.heads)

        # map_out = torch.einsum('bhji,bhjd->bhid', map_feat_attn, feat_v)
        # map_out = rearrange(map_out, 'b heads (h w) dim_head -> b (dim_head heads) h w', b=B, dim_head=self.dim_head, heads=self.heads, h=self.map_size, w=self.map_size)

        feat_out = self.proj_drop(self.feat_out(feat_out))
        # map_out = self.proj_drop(self.map_out(map_out))

        # return feat_out + res
        return feat_out


class up_block_multi(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=False, heads=4, attn_drop=0.):
        super().__init__()
        self.upbil = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cha = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        if bottleneck:
            block = None
            # block = BottleneckBlock
        else:
            block = BasicBlock
        # attn_block = AttentionBlockB
        self.attn = upAttention(out_ch,in_ch,out_ch,heads=heads,attn_drop=attn_drop,proj_drop=attn_drop)


        block_list = []
        block_list.append(block(2 * out_ch, out_ch, stride=1))
        # block_list.append(block(out_ch, out_ch, stride=1))
        # block_list.append(attn_block(dim_in=out_ch, dim_out=out_ch, heads=heads, sd=attn_drop))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        outbil = self.cha(x1)
        outbil = self.upbil(outbil)
        # outbil = self.cha(outbil)
        outconv = self.upconv(x1)
        outatt = self.attn(x1)
        out =outbil+outatt+outconv
        # out = self.fusion(x2, out)
        # x1: low-res 语义丰富 feature, x2: high-res feature
        # out = self.attn_decoder(x1, x2)
        out = torch.cat([out, x2], dim=1)
        out = self.blocks(out)
        return out


class BidirectionAttention(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads=4, dim_head=64, attn_drop=0.,
                 proj_drop=0., map_size=56, proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.GELU):
        super().__init__()

        # 更改map_dim维度
        self.map_projection = nn.Conv2d(map_dim, out_dim, kernel_size=1, bias=False)
        self.map_dim = out_dim

        self.inner_dim = dim_head * heads
        self.feat_dim = feat_dim

        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.map_size = map_size

        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.norm1 = norm(feat_dim) if norm else nn.Identity()  # norm layer for feature map
        self.norm2 = norm(self.map_dim) if norm else nn.Identity()  # norm layer for semantic map

        assert proj_type in ['linear', 'depthwise']

        if proj_type == 'linear':
            self.feat_qv = nn.Conv2d(feat_dim, self.inner_dim * 2, kernel_size=1, bias=False)
            self.feat_out = nn.Conv2d(self.inner_dim, out_dim, kernel_size=1, bias=False)

        else:
            self.feat_qv = depthwise_separable_conv(feat_dim, self.inner_dim * 2)
            self.feat_out = depthwise_separable_conv(self.inner_dim, out_dim)

        self.map_qv = nn.Conv2d(self.map_dim, self.inner_dim * 2, kernel_size=1, bias=False)
        # self.map_out = nn.Conv2d(self.inner_dim, map_dim, kernel_size=1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat, semantic_map):

        B, C, H, W = feat.shape
        B, C, Hh, Wh = feat.shape

        res = feat
        feat = self.norm1(feat)
        semantic_map = self.map_projection(semantic_map)
        semantic_map = self.norm2(semantic_map)

        feat_q, feat_v = self.feat_qv(feat).chunk(2, dim=1)  # B, inner_dim, H, W
        map_q, map_v = self.map_qv(semantic_map).chunk(2, dim=1)  # B, inner_dim, rs, rs

        feat_q, feat_v = map(
            lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                heads=self.heads, h=H, w=W), [feat_q, feat_v])
        map_q, map_v = map(
            lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                heads=self.heads, h=self.map_size, w=self.map_size), [map_q, map_v])
        #交叉注意 双q
        attn = torch.einsum('bhid,bhjd->bhij', feat_q, map_q)
        attn *= self.scale
        # 对map语义那维进行softmax
        feat_map_attn = F.softmax(attn, dim=-1)  # semantic map is very concise that don't need dropout
        # add dropout migth cause unstable during training
        # map_feat_attn = self.attn_drop(F.softmax(attn, dim=-2))

        feat_out = torch.einsum('bhij,bhjd->bhid', feat_map_attn, map_v)
        feat_out = rearrange(feat_out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W,
                             dim_head=self.dim_head, heads=self.heads)

        # map_out = torch.einsum('bhji,bhjd->bhid', map_feat_attn, feat_v)
        # map_out = rearrange(map_out, 'b heads (h w) dim_head -> b (dim_head heads) h w', b=B, dim_head=self.dim_head, heads=self.heads, h=self.map_size, w=self.map_size)

        feat_out = self.proj_drop(self.feat_out(feat_out))
        # map_out = self.proj_drop(self.map_out(map_out))

        return feat_out + res


class Chan_spaAtt(nn.Module):
    def __init__(self, channels=64, out_cha=64, r=4):
        super().__init__()
        inter_channels = int(channels // r)
        in_dim = channels
        self.chanel_in = in_dim

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_cha, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU())

    def forward(self, x):
        # 先对低级信息进行处理 residual上采样后
        # x = self.filter(x)
        xa = x
        xl = self.local_att(x)  # torch.Size([8, 64, 32, 32])
        xg = self.global_att(x)  # # torch.Size([8, 64, 1, 1])
        xlg = xl + xg  # torch.Size([8, 64, 32, 32])
        wei = self.sigmoid(xlg)  # torch.Size([8, 64, 32, 32])

        xo = x * wei

        m_batchsize, C, height, width = xo.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + xo
        # # 空间注意
        # xo = self.spa_att(xo)
        out = self.conv(xo)
        return out


########################################################################
# Transformer components

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, p, **kwargs):
        super().__init__()

        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)

        return x

    def extra_repr(self):
        return "p=%s" % repr(self.p)


class Attention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, k=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        # # (b n1 n2) c p1 p2
        b, n, _, y = x.shape
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)

        out = self.to_out(out)

        return out, attn


# from visualizer import get_local
class Attention2d_win(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, k=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

#      ----------window--------------
        self.window_tokens = nn.Parameter(torch.randn(dim_in))

        # prenorm and non-linearity for window tokens
        # then projection to queries and keys for window tokens

        self.window_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(inner_dim, inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h=heads),
        )

        # window attention

        self.window_attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    # @get_local('attn_map')
    def forward(self, x, mask=None):
        # b, n, _, y = x.shape  # (b n1 n2) c p1 p2
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p
        x = rearrange(x, "b c (n1 p1) (n2 p2) -> (b n1 n2) c (p1 p2)", p1=p, p2=p)

        num_windows = n1 * n2
        w = repeat(self.window_tokens, 'c -> b c 1', b = x.shape[0])
        x = torch.cat((w, x), dim=-1)
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)
        # attn_map =attn
        attn_maps.append(attn.clone().detach().cpu().numpy())

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        global_tokens, fmp = out[:, :, 0], out[:, :, 1:]
        if num_windows == 1:
            fmap = rearrange(fmp, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x = n1, y = n2, w1 = p, w2 = p)
            return self.to_out(fmap)
        window_tokens = rearrange(global_tokens, '(b x y) h d -> b h (x y) d', x=n1, y=n2)
        windowed_fmaps = rearrange(fmp, '(b x y) h n d -> b h (x y) n d', x=n1, y=n2)

        # windowed queries and keys (preceded by prenorm activation)
        # torch.Size([24, 8, 64, 32])
        w_q, w_k = self.window_tokens_to_qk(window_tokens).chunk(2, dim=-1)

        # scale

        w_q = w_q * self.scale
        # similarities
        # torch.Size([24, 8, 64, 64])
        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)
        # torch.Size([24, 8, 64, 64])
        w_attn = self.window_attend(w_dots)

        # aggregate the feature maps from the "depthwise" attention step (the most interesting part of the paper, one i haven't seen before)
        # torch.Size([24, 8, 64, 49, 32])
        aggregated_windowed_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, windowed_fmaps)

        # fold back the windows and then combine heads for aggregation
        # torch.Size([24, 256, 56, 56])
        fmap = rearrange(aggregated_windowed_fmap, 'b h (x y) (w1 w2) d -> b (h d) (x w1) (y w2)', x=n1,
                         y=n2, w1=p, w2=p)
        out = self.to_out(fmap)
        return out,attn

        # out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)
        #
        # out = self.to_out(out)

        # return out, attn

# mask +pos 0+rel dis
class LocalAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 window_size=7, k=1,
                 heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        self.attn = Attention2d(dim_in, dim_out,
                                heads=heads, dim_head=dim_head, dropout=dropout, k=k)
        self.window_size = window_size
        # torch.Size([49, 49, 2])
        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.rel_index = self.rel_index.type(torch.long)
        #  3,3,224,224 测试 torch.Size([13, 13])
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p

        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        # re =self.rel_index.type(torch.long)
        # print(type(re))
        # print(self.pos_embedding[re[:, :, 0],    x,y
        er = self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]
        print("---shap--",er.shape) # torch.Size([49, 49])
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]

        # x = rearrange(x, "b c (n1 p1) (n2 p2) -> (b n1 n2) c p1 p2", p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, "(b n1 n2) c p1 p2 -> b c (n1 p1) (n2 p2)", n1=n1, n2=n2, p1=p, p2=p)

        return x, attn

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        print("rel-----------------",i.shape)
        d = i[None, :, :] - i[:, None, :]

        return d



class AttentionBlockB(nn.Module):
    # Attention block with pre-activation.
    expansion = 1

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, sd=0.0,
                 stride=1, window_size=7, k=1, norm=nn.BatchNorm2d, activation=nn.GELU,
                 **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        # attn = partial(LocalAttention, window_size=window_size, k=k)
        attn = Attention2d_win(dim_in, dim_out,
                                heads=heads, dim_head=dim_head, dropout=dropout, k=k)
        width = dim_in // self.expansion

        self.shortcut = []
        if stride != 1 or dim_in != dim_out * self.expansion:
            self.shortcut.append(conv1x1(dim_in, dim_out * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.norm1 = norm(dim_in)
        self.relu = activation()

        self.conv = nn.Conv2d(dim_in, width, kernel_size=1, bias=False)
        self.norm2 = norm(width)
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.norm1(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.conv(x)
        x = self.norm2(x)
        x, attn = self.attn(x)

        x = self.sd(x) + skip

        return x








class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        #self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim*3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            #self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):

        B, C, H, W = x.shape

        #B, inner_dim, H, W
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # 使k,v大小匹配到self.reduce_size大小  有效自注意
        # print(self.projection, self.reduce_size)
        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        # 变换维度顺序类似
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=H, w=W)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        # 矩阵乘法
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)
            q_k_attn += relative_position_bias
            #rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            #q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn

# 两个特征自注意  对低分辨率特征进行加权
class LinearAttentionDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        #self.to_kv = nn.Conv2d(dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_q = nn.Conv2d(dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim*2)
        self.to_q = depthwise_separable_conv(out_dim, self.inner_dim)
        self.to_out = depthwise_separable_conv(self.inner_dim, out_dim)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            #self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, q, x):

        B, C, H, W = x.shape # low-res feature shape
        BH, CH, HH, WH = q.shape # high-res feature shape

        # low-res feature 求k,v 最后对低特征进行特征加重
        k, v = self.to_kv(x).chunk(2, dim=1) #B, inner_dim, H, W
        q = self.to_q(q) #BH, inner_dim, HH, WH

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))

        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=HH, w=WH)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(HH, WH)
            q_k_attn += relative_position_bias
            #rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, HH, WH, self.dim_head)
            #q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn

class RelativePositionEmbedding(nn.Module):
    # input-dependent relative position
    def __init__(self, dim, shape):
        super().__init__()

        self.dim = dim
        self.shape = shape

        self.key_rel_w = nn.Parameter(torch.randn((2*self.shape-1, dim))*0.02)
        self.key_rel_h = nn.Parameter(torch.randn((2*self.shape-1, dim))*0.02)

        coords = torch.arange(self.shape)
        relative_coords = coords[None, :] - coords[:, None] # h, h
        relative_coords += self.shape - 1 # shift to start from 0

        self.register_buffer('relative_position_index', relative_coords)



    def forward(self, q, Nh, H, W, dim_head):
        # q: B, Nh, HW, dim
        B, _, _, dim = q.shape

        # q: B, Nh, H, W, dim_head
        q = rearrange(q, 'b heads (h w) dim_head -> b heads h w dim_head', b=B, dim_head=dim_head, heads=Nh, h=H, w=W)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, 'w')

        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.key_rel_h, 'h')

        return rel_logits_w, rel_logits_h

    def relative_logits_1d(self, q, rel_k, case):

        B, Nh, H, W, dim = q.shape

        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k) # B, Nh, H, W, 2*shape-1

        if W != self.shape:
            # self_relative_position_index origin shape: w, w
            # after repeat: W, w
            relative_index= torch.repeat_interleave(self.relative_position_index, W//self.shape, dim=0) # W, shape
        relative_index = relative_index.view(1, 1, 1, W, self.shape)
        relative_index = relative_index.repeat(B, Nh, H, 1, 1)

        rel_logits = torch.gather(rel_logits, 4, relative_index) # B, Nh, H, W, shape
        rel_logits = rel_logits.unsqueeze(3)
        rel_logits = rel_logits.repeat(1, 1, 1, self.shape, 1, 1)

        if case == 'w':
            rel_logits = rearrange(rel_logits, 'b heads H h W w -> b heads (H W) (h w)')

        elif case == 'h':
            rel_logits = rearrange(rel_logits, 'b heads W w H h -> b heads (H W) (h w)')

        return rel_logits




class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    # Borrowed some code from SwinTransformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
                torch.randn((2*h-1) * (2*w-1), num_heads)*0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, h, w
        coords_flatten = torch.flatten(coords, 1) # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1) # hw, hw

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h, self.w, self.h*self.w, -1) #h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H//self.h, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W//self.w, dim=1) #HW, hw, nH

        relative_position_bias_expanded = relative_position_bias_expanded.view(H*W, self.h*self.w, self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)

        return relative_position_bias_expanded


###########################################################################
# Unet Transformer building block
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.register_buffer()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# 空间注意模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# 融合模块
class AFF(nn.Module):

    def __init__(self, channels=64, r=4, ince_bool=True, ince_ch=[16, 32, 64, 128, 32, 32]):
        super(AFF, self).__init__()
        # print("-----aff---------",channels,r)
        inter_channels = int(channels // r)
        # depthwise实现滤波
        self.filter = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, stride=1)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

        # 新加模块---空间注意
        if ince_bool:
            self.spa_att = Inception(channels, ince_ch[0], ince_ch[1], ince_ch[2], ince_ch[3], ince_ch[4], ince_ch[5])
        else:
            self.spa_att = SpatialAttention()

    def forward(self, x, residual):
        # 先对低级信息进行处理
        x = self.filter(x)
        xa = x + residual
        xl = self.local_att(xa)  # torch.Size([8, 64, 32, 32])
        xg = self.global_att(xa)  # # torch.Size([8, 64, 1, 1])
        xlg = xl + xg  # torch.Size([8, 64, 32, 32])
        wei = self.sigmoid(xlg)  # torch.Size([8, 64, 32, 32])

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        # 空间注意
        xo = self.spa_att(xo) * xo

        return xo


#  残差卷积 加vit卷积 看是否下采样maxpool 先maxpool 再卷积再vit
class down_block_trans(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=False, maxpool=True, num_croblock=0,
                 heads=4, attn_drop=0.0, trans=False):

        super().__init__()

        block_list = []

        if bottleneck:
            block = None
            # block = BottleneckBlock
        else:
            block = BasicBlock

        attn_block = AttentionBlockB
        gl_attblock = BasicTransBlock

        if maxpool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch, stride=1))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        # block_list.append(attn_block(out_ch, out_ch, heads=heads, sd=attn_drop))
        # if trans:
        #     # BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan // num_heads[-5],
        #     #                                             attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
        #     #                                             projection=projection, rel_pos=rel_pos)
        #
        #     # attn_block(out_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop,
        #     # reduce_size=reduce_size, projection=projection, rel_pos
        #     block_list.append(gl_attblock(out_ch, heads,))

        for i in range(num_croblock):
            block_list.append(block(out_ch, out_ch, stride=1))
            block_list.append(attn_block(out_ch, out_ch, heads=heads, sd=attn_drop))

        self.blocks = nn.Sequential(*block_list)

        
    def forward(self, x):
        
        out = self.blocks(x)
        return out

# 先自注意融合 再vit卷积再残差卷积
class up_block_trans(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=False,  bilinear=False,heads=4,attn_drop=0.):
        super().__init__()
        # self.fusion = AFF(out_ch, ince_bool=False)
        # self.attn_decoder = BasicTransDecoderBlock(in_ch, out_ch, heads=heads, dim_head=dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.cha = nn.Conv2d(in_ch,out_ch,kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        if bottleneck:
            block = None
            # block = BottleneckBlock
        else:
            block = BasicBlock
        attn_block = AttentionBlockB
        
        block_list = []
        block_list.append(block(2*out_ch, out_ch, stride=1))
        # block_list.append(block(out_ch, out_ch, stride=1))
        # block_list.append(attn_block(dim_in=out_ch, dim_out=out_ch, heads=heads, sd=attn_drop))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        out = self.up(x1)
        # out = self.fusion(x2, out)
        # x1: low-res 语义丰富 feature, x2: high-res feature
        # out = self.attn_decoder(x1, x2)
        out = torch.cat([out, x2], dim=1)
        out = self.blocks(out)
        return out


class up_block_transADj(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=False, bilinear=False, heads=4, attn_drop=0.):
        super().__init__()
        # self.fusion = AFF(out_ch, ince_bool=False)

        # self.attn_decoder = BasicTransDecoderBlock(in_ch, out_ch, heads=heads, dim_head=dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.cha = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.att = BidirectionAttention(out_ch, 64, out_ch, heads=heads, dim_head=32, attn_drop=attn_drop,
                                        proj_drop=attn_drop)
        if bottleneck:
            block = None
            # block = BottleneckBlock
        else:
            block = BasicBlock
        attn_block = AttentionBlockB

        block_list = []
        block_list.append(block(2 * out_ch, out_ch, stride=1))
        # block_list.append(block(out_ch, out_ch, stride=1))
        # block_list.append(attn_block(dim_in=out_ch, dim_out=out_ch, heads=heads, sd=attn_drop))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1, x2,fuse):
        out = self.up(x1)
        if fuse is not None:
            print("---out_shap---",out.shape)
            out = self.att(out,fuse)

        # out = self.fusion(x2, out)
        # x1: low-res 语义丰富 feature, x2: high-res feature
        # out = self.attn_decoder(x1, x2)
        out = torch.cat([out, x2], dim=1)
        out = self.blocks(out)
        return out


# 重复多少次BasicTransBlock
class block_trans(nn.Module):
    def __init__(self, in_ch, num_block, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):

        super().__init__()

        block_list = []

        attn_block = BasicTransBlock

        assert num_block > 0
        for i in range(num_block):
            block_list.append(attn_block(in_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos))
        self.blocks = nn.Sequential(*block_list)

        
    def forward(self, x):
        
        out = self.blocks(x)


        return out


class BidirectionAttention(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads=4, dim_head=64, attn_drop=0.,
                 proj_drop=0., map_size=8, proj_type='depthwise'):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.feat_dim = feat_dim
        self.map_dim = map_dim
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.map_size = map_size

        assert proj_type in ['linear', 'depthwise']

        if proj_type == 'linear':
            self.feat_qv = nn.Conv2d(feat_dim, self.inner_dim * 2, kernel_size=1, bias=False)
            self.feat_out = nn.Conv2d(self.inner_dim, out_dim, kernel_size=1, bias=False)

        else:
            self.feat_qv = depthwise_separable_conv(feat_dim, self.inner_dim * 2)
            self.feat_out = depthwise_separable_conv(self.inner_dim, out_dim)

        self.map_qv = nn.Conv2d(map_dim, self.inner_dim * 2, kernel_size=1, bias=False)
        self.map_out = nn.Conv2d(self.inner_dim, map_dim, kernel_size=1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat, semantic_map):

        B, C, H, W = feat.shape

        feat_q, feat_v = self.feat_qv(feat).chunk(2, dim=1)  # B, inner_dim, H, W
        map_q, map_v = self.map_qv(semantic_map).chunk(2, dim=1)  # B, inner_dim, rs, rs

        feat_q, feat_v = map(
            lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                heads=self.heads, h=H, w=W), [feat_q, feat_v])
        map_q, map_v = map(
            lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                heads=self.heads, h=self.map_size, w=self.map_size), [map_q, map_v])

        attn = torch.einsum('bhid,bhjd->bhij', feat_q, map_q)
        attn *= self.scale
        # 对map语义那维进行softmax
        feat_map_attn = F.softmax(attn, dim=-1)  # semantic map is very concise that don't need dropout
        # add dropout migth cause unstable during training
        map_feat_attn = self.attn_drop(F.softmax(attn, dim=-2))

        feat_out = torch.einsum('bhij,bhjd->bhid', feat_map_attn, map_v)
        feat_out = rearrange(feat_out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W,
                             dim_head=self.dim_head, heads=self.heads)

        map_out = torch.einsum('bhji,bhjd->bhid', map_feat_attn, feat_v)
        map_out = rearrange(map_out, 'b heads (h w) dim_head -> b (dim_head heads) h w', b=B, dim_head=self.dim_head,
                            heads=self.heads, h=self.map_size, w=self.map_size)

        feat_out = self.proj_drop(self.feat_out(feat_out))
        map_out = self.proj_drop(self.map_out(map_out))

        return feat_out, map_out


class BidirectionAttentionBlock(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads, dim_head, norm=nn.BatchNorm2d,
                 act=nn.GELU, expansion=4, attn_drop=0., proj_drop=0., map_size=8,
                 proj_type='depthwise'):
        super().__init__()

        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]
        assert proj_type in ['linear', 'depthwise']

        self.norm1 = norm(feat_dim) if norm else nn.Identity()  # norm layer for feature map
        self.norm2 = norm(map_dim) if norm else nn.Identity()  # norm layer for semantic map

        self.attn = BidirectionAttention(feat_dim, map_dim, out_dim, heads=heads, dim_head=dim_head,
                                         attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size,
                                         proj_type=proj_type)

        self.shortcut = nn.Sequential()
        # if feat_dim != out_dim:
        #     self.shortcut = ConvNormAct(feat_dim, out_dim, kernel_size=1, padding=0, norm=norm, act=act, preact=True)
        #
        # if proj_type == 'linear':
        #     self.feedforward = FusedMBConv(out_dim, out_dim, expansion=expansion, kernel_size=1, act=act,
        #                                    norm=norm)  # 2 conv1x1
        # else:
        #     self.feedforward = MBConv(out_dim, out_dim, expansion=expansion, kernel_size=3, act=act, norm=norm,
        #                               p=proj_drop)  # depthwise conv

    def forward(self, x, semantic_map):

        feat = self.norm1(x)
        mapp = self.norm2(semantic_map)

        out, mapp = self.attn(feat, mapp)

        out += self.shortcut(x)
        # out = self.feedforward(out)
        #
        # mapp += semantic_map

        return out, mapp

class up_block_deform(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=False, heads=4, attn_drop=0.,sr_ratio=8):
        super().__init__()

        # self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.gt = BasicTransDE(in_ch, out_ch, heads, attn_drop=attn_drop,sr_ratio=sr_ratio)
        self.deform_att = BasicTransDecoderBlock(in_ch, out_ch, heads=heads, dim_head=dim_head, attn_drop=attn_drop,
                                                   proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                                   rel_pos=rel_pos)

        if bottleneck:
            block = None
            # block = BottleneckBlock
        else:
            block = BasicBlock
        # attn_block = AttentionBlockB

        block_list = []
        block_list.append(block(2 * out_ch, out_ch, stride=1))
        # block_list.append(attn_block(dim_in=out_ch, dim_out=out_ch, heads=heads, sd=attn_drop))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        # x_ad = self.conv(x1)  # 调整通道
        x_o = self.deform_att(x1, x2)  # 对低级语义特征进行自注意调整   #多尺度上采样操作

        out = torch.cat([x_o, x2], dim=1)
        out = self.blocks(out)
        return out

class AttentionDepsp(nn.Module):

    def __init__(self, in_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., projection='interp',reduce_size=16):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        # self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        #self.to_kv = nn.Conv2d(dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_q = nn.Conv2d(dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim*2)
        self.to_q = depthwise_separable_conv(out_dim, self.inner_dim)
        self.to_out = depthwise_separable_conv(self.inner_dim, out_dim)

        self.psp = PSPModule(sizes=(1,3,6,8))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # if self.rel_pos:
        #     self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
        #     #self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, q, x):  # x2,x1----------大---小

        B, C, H, W = x.shape # low-res feature shape  256,28,28原  后high
        BH, CH, HH, WH = q.shape # high-res feature shape 512,14,14原  后low

        # low-res feature 求k,v 最后对低特征进行特征加重
        k, v = self.to_kv(x).chunk(2, dim=1) #B, inner_dim, H, W
        k = self.psp(k)
        _,_,d = k.shape
        # k =k.view(B,self.inner_dim,int(d**0.5),-1)
        v = self.psp(v)

        q = self.to_q(q) #BH, inner_dim, HH, WH



        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=HH, w=WH)
        # k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) d -> b heads d dim_head', dim_head=self.dim_head, heads=self.heads,), (k, v))

        # k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=H, w=W), (k, v))


        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)


        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)
        # out +=q
        return out, q_k_attn


class BasicTransDEpsp(nn.Module):

    def __init__(self, in_ch, out_ch, heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        # self.window_size = windowsize
        # self.se=dwSEBlock(out_ch*2,out_ch)
        self.bn_l = nn.BatchNorm2d(in_ch)
        self.bn_h = nn.BatchNorm2d(out_ch)

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.attn = AttentionDepsp(in_ch, out_ch, heads=heads, dim_head=out_ch // heads, attn_drop=attn_drop,
                                proj_drop=proj_drop)
        # self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # self.gatefus = nn.Conv2d(3*out_ch,3,3,1,1,bias=True)

        # self.bn2 = nn.BatchNorm2d(out_ch)
        # self.relu = nn.ReLU(inplace=True)
        # self.mlp = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape  # low-res feature shape  256,28,28原  后high
        BH, CH, HH, WH = x2.shape
        # p = self.window_size
        # n1 = H // p
        # n2 = W // p
        # nn1 = HH // p
        # nn2 = WH// p
        # x1 = rearrange(x1, "b c (n1 p1) (n2 p2) -> (b n1 n2) c p1 p2", p1=p, p2=p)
        # x2 = rearrange(x2, "b c (nn1 p1) (nn2 p2)-> (b nn1 nn2) c p1 p2", p1=p, p2=p)

        # 低特征上采样
        # res = x2
        # x11 = self.conv_ch(x1)
        residue = F.interpolate(self.conv_ch(x1), size=x2.shape[-2:], mode='bilinear', align_corners=True)
        # x1: low-res, x2: high-res
        x1 = self.bn_l(x1)  # x1----512,14,14
        x2 = self.bn_h(x2)  # x2----256,28,28
        # 主要使用低分辨率特征
        out, q_k_attn = self.attn(x2, x1)  # q,x----- x2--k,v--psp采样
        # out = rearrange(out, "(b nn1 nn2) c p1 p2 -> b c (nn1 p1) (nn2 p2)", nn1=nn1, nn2=nn2, p1=p, p2=p)
        # upconv = self.up(x1)
        # gate = self.gatefus(torch.cat((out,upconv,residue),dim=1))

        # 原先x2,x1 output与x2大小一样 更大尺度的那个 与query一样
        # 现输出
        #
        # out = out + res
        # out = torch.cat((out,residue),dim=1)
        # out = self.se(out)
        # out = out + residue  #原来bil+att
        # out = out*gate[:,[0],:,:] + residue*gate[:,[1],:,:] + upconv*gate[:,[2],:,:]
        # residue = out

        # out = self.bn2(out)
        # out = self.relu(out)
        # out = self.mlp(out)

        out += residue

        return out

class up_block_transpsp(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=False, heads=4, attn_drop=0.,sr_ratio=8):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.gt = BasicTransDEpsp(in_ch, out_ch, heads, attn_drop=attn_drop,proj_drop=attn_drop)
        if bottleneck:
            block = None
            # block = BottleneckBlock
        else:
            block = BasicBlock
        # attn_block = AttentionBlockB

        block_list = []
        block_list.append(block(2 * out_ch, out_ch, stride=1))
        # block_list.append(attn_block(dim_in=out_ch, dim_out=out_ch, heads=heads, sd=attn_drop))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        # x_ad = self.conv(x1)  # 调整通道
        x_o = self.gt(x1, x2)  # 对低级语义特征进行自注意调整   #多尺度上采样操作
        # out = self.up(x1)
        # x1: low-res 语义丰富 feature, x2: high-res feature
        # out = self.attn_decoder(x1, x2)
        # out = self.fusion(x2,out)
        out = torch.cat([x_o, x2], dim=1)
        out = self.blocks(out)
        return out



class up_block_transGT(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=False, heads=4, attn_drop=0.,sr_ratio=8):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.gt = BasicTransDE(in_ch, out_ch, heads, attn_drop=attn_drop,sr_ratio=sr_ratio)
        if bottleneck:
            block = None
            # block = BottleneckBlock
        else:
            block = BasicBlock
        # attn_block = AttentionBlockB

        block_list = []
        block_list.append(block(2 * out_ch, out_ch, stride=1))
        # block_list.append(attn_block(dim_in=out_ch, dim_out=out_ch, heads=heads, sd=attn_drop))
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        # x_ad = self.conv(x1)  # 调整通道
        x_o = self.gt(x1, x2)  # 对低级语义特征进行自注意调整   #多尺度上采样操作
        # out = self.up(x1)
        # x1: low-res 语义丰富 feature, x2: high-res feature
        # out = self.attn_decoder(x1, x2)
        # out = self.fusion(x2,out)
        out = torch.cat([x_o, x2], dim=1)
        out = self.blocks(out)
        return out


class BasicTransDE(nn.Module):

    def __init__(self, in_ch, out_ch, heads, attn_drop=0., proj_drop=0., sr_ratio=7):
        super().__init__()
        # self.window_size = windowsize

        self.bn_l = nn.BatchNorm2d(in_ch)
        self.bn_h = nn.BatchNorm2d(out_ch)

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.attn = AttentionDe(in_ch, out_ch, heads=heads, dim_head=out_ch // heads, attn_drop=attn_drop,
                                proj_drop=proj_drop,sr_ratio=sr_ratio)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # self.gatefus = nn.Conv2d(3 * out_ch, 3, 3, 1, 1, bias=True)

        # self.bn2 = nn.BatchNorm2d(out_ch)
        # self.relu = nn.ReLU(inplace=True)
        # self.mlp = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape  # low-res feature shape  256,28,28原  后high
        BH, CH, HH, WH = x2.shape

        residue = F.interpolate(self.conv_ch(x1), size=x2.shape[-2:], mode='bilinear', align_corners=True)
        # x1: low-res, x2: high-res
        x1 = self.bn_l(x1)  # x1----512,14,14
        x2 = self.bn_h(x2)  # x2----256,28,28
        # 主要使用低分辨率特征
        out, q_k_attn = self.attn(x2, x1)  # q,x----- x2--k,v--psp采样
        # out = rearrange(out, "(b nn1 nn2) c p1 p2 -> b c (nn1 p1) (nn2 p2)", nn1=nn1, nn2=nn2, p1=p, p2=p)
        upconv = self.up(x1)
        # gate = self.gatefus(torch.cat((out, upconv, residue), dim=1))

        # 原先x2,x1 output与x2大小一样 更大尺度的那个 与query一样
        # 现输出
        #
        # out = out + res
        out = out + residue  #原来bil+att
        # out = out * gate[:, [0], :, :] + residue * gate[:, [1], :, :] + upconv * gate[:, [2], :, :]
        # residue = out

        # out = self.bn2(out)
        # out = self.relu(out)
        # out = self.mlp(out)

        # out += residue

        return out
#layer_norm
class AttentionDe_lm(nn.Module):

    def __init__(self, in_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., projection='interp',reduce_size=16,sr_ratio=8):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        # self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        #self.to_kv = nn.Conv2d(dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_q = nn.Conv2d(dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim*2)
        self.to_q = depthwise_separable_conv(out_dim, self.inner_dim)
        self.to_out = depthwise_separable_conv(self.inner_dim, out_dim)

        # self.psp = PSPModule(sizes=(1,3,6,8))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.conv = nn.Conv2d(self.inner_dim*2, self.inner_dim, kernel_size=1, stride=1)
            self.act = nn.GELU()
            if sr_ratio == 16:
                self.sr1 = nn.Conv2d(in_dim, in_dim, kernel_size=16, stride=16)
                self.norm1 = nn.LayerNorm(in_dim)
                self.sr2 = nn.Conv2d(in_dim, in_dim, kernel_size=8, stride=8)
                self.norm2 = nn.LayerNorm(in_dim)
            if sr_ratio == 8:
                self.sr1 = nn.Conv2d(in_dim, in_dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(in_dim)
                self.sr2 = nn.Conv2d(in_dim, in_dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(in_dim)
            if sr_ratio == 4:
                self.sr1 = nn.Conv2d(in_dim, in_dim, kernel_size=4, stride=4)
                self.norm1 = nn.LayerNorm(in_dim)
                self.sr2 = nn.Conv2d(in_dim,in_dim, kernel_size=2, stride=2)
                self.norm2 = nn.LayerNorm(in_dim)
            if sr_ratio == 2:
                self.sr1 = nn.Conv2d(in_dim,in_dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(in_dim)
                self.sr2 = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(in_dim)
                # self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                # self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
                self.to_kv1 = depthwise_separable_conv(in_dim, self.inner_dim * 2)
                self.to_kv2 = depthwise_separable_conv(in_dim, self.inner_dim * 2)
                # self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
                # self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
            else:
                # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
                self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim * 2)
                # self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            # self.apply(self._init_weights)



    def forward(self, q, x):  # x2,x1----------大---小

        B, C, H, W = x.shape # low-res feature shape  256,28,28原  后high
        BH, CH, HH, WH = q.shape # high-res feature shape 512,14,14原  后low
        q = self.to_q(q)  # BH, inner_dim, HH, WH
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=HH, w=WH)

        # low-res feature 求k,v 最后对低特征进行特征加重
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1]  # B head N C
            k2, v2 = kv2[0], kv2[1]
            # x_1 = self.act(self.norm1(self.sr1(x)))
            # _,_,h1,w1 = x_1.shape
            # x_2 = self.act(self.norm2(self.sr2(x)))
            # _, _, h2, w2 = x_2.shape
            # k1,v1 = self.to_kv1(x_1).chunk(2, dim=1) #B, inner_dim, H, W
            # k1, v1 = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=h1, w=w1), (k1, v1))
            # k2, v2 = self.to_kv2(x_2).chunk(2, dim=1)  # B, inner_dim, H, W
            # k2, v2 = map(
            #     lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
            #                         heads=self.heads, h=h2, w=w2), (k2, v2))


            attn1 = (q @ k1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
            #                            transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
            #     view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            # x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)
            x1 = torch.einsum('bhij,bhjd->bhid', attn1, v1)
            x1 = rearrange(x1, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head,
                            heads=self.heads)

            attn2 = (q @ k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).
            #                            transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \
            #     view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            # x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)
            x2 = torch.einsum('bhij,bhjd->bhid', attn2, v2)
            x2 = rearrange(x2, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head,
                           heads=self.heads)
            out = torch.cat([x1, x2], dim=1)
            out =self.conv(out)
        else:
            # kv = self.to_kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # k, v = kv[0], kv[1]
            k, v = self.to_kv(x).chunk(2, dim=1) #B, inner_dim, H, W
            k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=H, w=W), (k, v))

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = torch.einsum('bhij,bhjd->bhid', attn, v)
            out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head,
                            heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out
# batch_norm
class AttentionDe(nn.Module):

    def __init__(self, in_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., projection='interp',reduce_size=16,sr_ratio=8):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        # self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        #self.to_kv = nn.Conv2d(dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_q = nn.Conv2d(dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim*2)
        self.to_q = depthwise_separable_conv(out_dim, self.inner_dim)
        self.to_out = depthwise_separable_conv(self.inner_dim, out_dim)

        # self.psp = PSPModule(sizes=(1,3,6,8))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.conv = nn.Conv2d(self.inner_dim*2, self.inner_dim, kernel_size=1, stride=1)
            self.act = nn.GELU()
            if sr_ratio == 16:
                self.sr1 = nn.Conv2d(in_dim, in_dim, kernel_size=16, stride=16)
                self.norm1 = nn.BatchNorm2d(in_dim)
                self.sr2 = nn.Conv2d(in_dim, in_dim, kernel_size=8, stride=8)
                self.norm2 = nn.BatchNorm2d(in_dim)
            if sr_ratio == 8:
                self.sr1 = nn.Conv2d(in_dim, in_dim, kernel_size=8, stride=8)
                self.norm1 = nn.BatchNorm2d(in_dim)
                self.sr2 = nn.Conv2d(in_dim, in_dim, kernel_size=4, stride=4)
                self.norm2 = nn.BatchNorm2d(in_dim)
            if sr_ratio == 4:
                self.sr1 = nn.Conv2d(in_dim, in_dim, kernel_size=4, stride=4)
                self.norm1 = nn.BatchNorm2d(in_dim)
                self.sr2 = nn.Conv2d(in_dim,in_dim, kernel_size=2, stride=2)
                self.norm2 = nn.BatchNorm2d(in_dim)
            if sr_ratio == 2:
                self.sr1 = nn.Conv2d(in_dim,in_dim, kernel_size=2, stride=2)
                self.norm1 = nn.BatchNorm2d(in_dim)
                self.sr2 = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1)
                self.norm2 = nn.BatchNorm2d(in_dim)
                # self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                # self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
                self.to_kv1 = depthwise_separable_conv(in_dim, self.inner_dim * 2)
                self.to_kv2 = depthwise_separable_conv(in_dim, self.inner_dim * 2)
                # self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
                # self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
            else:
                # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
                self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim * 2)
                # self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            # self.apply(self._init_weights)



    def forward(self, q, x):  # x2,x1----------大---小

        B, C, H, W = x.shape # low-res feature shape  256,28,28原  后high
        BH, CH, HH, WH = q.shape # high-res feature shape 512,14,14原  后low
        q = self.to_q(q)  # BH, inner_dim, HH, WH
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=HH, w=WH)

        # low-res feature 求k,v 最后对低特征进行特征加重
        if self.sr_ratio > 1:
            # x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_1 = self.act(self.norm1(self.sr1(x)))
            _,_,h1,w1 = x_1.shape
            x_2 = self.act(self.norm2(self.sr2(x)))
            _, _, h2, w2 = x_2.shape
            k1,v1 = self.to_kv1(x_1).chunk(2, dim=1) #B, inner_dim, H, W
            k1, v1 = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=h1, w=w1), (k1, v1))
            k2, v2 = self.to_kv2(x_2).chunk(2, dim=1)  # B, inner_dim, H, W
            k2, v2 = map(
                lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                    heads=self.heads, h=h2, w=w2), (k2, v2))


            attn1 = (q @ k1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
            #                            transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
            #     view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            # x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)
            x1 = torch.einsum('bhij,bhjd->bhid', attn1, v1)
            x1 = rearrange(x1, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head,
                            heads=self.heads)

            attn2 = (q @ k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).
            #                            transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \
            #     view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            # x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)
            x2 = torch.einsum('bhij,bhjd->bhid', attn2, v2)
            x2 = rearrange(x2, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head,
                           heads=self.heads)
            out = torch.cat([x1, x2], dim=1)
            out =self.conv(out)
        else:
            # kv = self.to_kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # k, v = kv[0], kv[1]
            k, v = self.to_kv(x).chunk(2, dim=1) #B, inner_dim, H, W
            k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=H, w=W), (k, v))

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = torch.einsum('bhij,bhjd->bhid', attn, v)
            out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head,
                            heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center



if __name__ == '__main__':
    att = AttentionBlockB(dim_in=32, dim_out=64, heads=8, sd=0.)
    inp = torch.rand((24,32,56,56))
    ouw = att(inp)
    print("-----",ouw.shape)


