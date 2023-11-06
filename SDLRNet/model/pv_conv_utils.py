
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import pdb
import einops
import numpy as np

from functools import partial

from einops.layers.torch import Rearrange
from torch import nn, einsum
from timm.models.layers import to_2tuple, trunc_normal_

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)



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

class DropPath(nn.Module):

    def __init__(self, p, **kwargs):
        super().__init__()

        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)

        return x

    def extra_repr(self):
        return "p=%s" % repr(self.p)

#######################################pvT#####################
class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize // 2, stride=1,
                              groups=hidden_features) ##相比mlp 多了这个卷积操作 利用深度卷积DWconv
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.fc2(x)
        # x.reshape(B, C, -1).permute(0, 2, 1)  ###原
        return x



class Glo_skipAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[1, 2, 3, 6]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.gloscale = dim ** -0.5

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        # self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        # self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        #########原先###################
        # self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        # self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        ###########后来#############
        self.qv_h = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.out_h = depthwise_separable_conv(dim,dim)
        self.qv_l = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.out_l = depthwise_separable_conv(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim) ##原----
        self.hlnorm = nn.LayerNorm(dim)
        # self.norm = nn.BatchNorm2d(dim)  # 后
        # window tokens

        self.global_tokensh = nn.Parameter(torch.randn(dim))
        self.global_tokensl = nn.Parameter(torch.randn(dim))

        # prenorm and non-linearity for window tokens
        # then projection to queries and keys for window tokens

        self.global_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(1), #补充的global——token为[3,64,1]
            nn.GELU(),# [3,2,32,1]
            Rearrange('b h n c -> b (h n) c'),
            nn.Conv1d(dim, dim * 2, 1),
            Rearrange('b (h n) c -> b h n c', h=num_heads),
        )
        self.hlglobal_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(1),  # 补充的global——token为[3,64,1]
            nn.GELU(),  # [3,2,32,1]
            Rearrange('b h n c -> b (h n) c'),
            nn.Conv1d(dim, dim * 2, 1),
            Rearrange('b (h n) c -> b h n c', h=num_heads),
        )

        # window attention

        self.global_attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(proj_drop)
        )
        # self.hlglobal_attend = nn.Sequential(
        #     nn.Softmax(dim=-1),
        #     nn.Dropout(proj_drop)
        # )

        # self.to_out = nn.Sequential(
        #     nn.Conv2d(dim, dim, 1),
        #     nn.Dropout(proj_drop)
        # )

    def forward(self, h_fea, l_fea, d_convs=None, d_convsl=None):
        # B, N, C = x.shape  # [3, 3136, 64]
        B, C, H, W = h_fea.shape
        Bl, Cl, Hl, Wl = l_fea.shape
        # B, inner_dim, H, W
        # qkv = self.to_qkv(x)
        # q, k, v = qkv.chunk(3, dim=1)

        q_h,v_h = self.qv_h(h_fea).chunk(2,dim=1)    # torch.Size([3, 64, 56, 56])
        # print("------------q_h-----------------",q_h.shape)
        q_l, v_l = self.qv_l(l_fea).chunk(2, dim=1)  #torch.Size([3, 64, 56, 56])
        q_h1, v_h1 = map(
            lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=C // self.num_heads,
                                heads=self.num_heads, h=H, w=W), [q_h, v_h])  # torch.Size([3, 2, 3136(56x56), 32])
        q_l1, v_l1 = map(
            lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=C // self.num_heads,
                                heads=self.num_heads, h=Hl, w=Wl), [q_l, v_l])  # torch.Size([3, 2, 3136, 32])

        # q1 = q.transpose(-2,-1)
        glo = repeat(self.global_tokensl, 'c-> b c 1 ', b=B)  # torch.Size([3, 64, 1])
        glo = rearrange(glo, 'b (dim_head heads) 1 -> b heads 1 dim_head', dim_head=C // self.num_heads,
                        heads=self.num_heads,
                        )  #torch.Size([3, 2, 1, 32])
        # glo1 = glo.transpose(-2,-1)  #torch.Size([3, 2, 32, 1])
        q_gl = torch.cat((glo, q_l1), dim=2)  # torch.Size([3, 2, 3137, 32])

        gloh = repeat(self.global_tokensh, 'c-> b c 1 ', b=B)  # torch.Size([3, 64, 1])
        gloh = rearrange(gloh, 'b (dim_head heads) 1 -> b heads 1 dim_head', dim_head=C // self.num_heads,
                        heads=self.num_heads,
                        )  # torch.Size([3, 2, 1, 32])
        # glo1 = glo.transpose(-2,-1)  #torch.Size([3, 2, 32, 1])
        q_gh = torch.cat((gloh, q_h1), dim=2)  # torch.Size([3, 2, 3137, 32])

        pools_h = []
        poolvs_h = []
        pools_l = []
        poolvs_l = []

        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            pool = F.adaptive_avg_pool2d(q_h, (round(H / pool_ratio), round(W / pool_ratio)))
            #原尺寸56--ratio--12--torch.Size([3, 64, 5, 5]) ratio--16--torch.Size([3, 64, 4, 4])
            # ratio--20--torch.Size([3, 64, 3, 3]) ratio--24--torch.Size([3, 64, 2, 2])
            #
            pool = pool + l(pool)  # 论文中说是相对位置编码  利用卷积
            # fix backward bug in higher torch versions when training
            # fix backward bug in higher torch versions when training
            pools_h.append(pool.view(B, C, -1))  #####原
            # pools.append(pool)  #####后
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            poolv = F.adaptive_avg_pool2d(v_h, (round(H / pool_ratio), round(W / pool_ratio)))
            poolv = poolv + l(poolv)  # 论文中说是相对位置编码  利用卷积
            # fix backward bug in higher torch versions when training
            poolvs_h.append(poolv.view(B, C, -1))  #####原
            # pools.append(pool)  #####后

        for (pool_ratio, l) in zip(self.pool_ratios, d_convsl):
            pool = F.adaptive_avg_pool2d(q_l, (round(H / pool_ratio), round(W / pool_ratio)))
            #原尺寸56--ratio--12--torch.Size([3, 64, 5, 5]) ratio--16--torch.Size([3, 64, 4, 4])
            # ratio--20--torch.Size([3, 64, 3, 3]) ratio--24--torch.Size([3, 64, 2, 2])
            pool = pool + l(pool)  # 论文中说是相对位置编码  利用卷积
            # fix backward bug in higher torch versions when training
            pools_l.append(pool.view(B, C, -1))  #####原
            # pools.append(pool)  #####后
        for (pool_ratio, l) in zip(self.pool_ratios, d_convsl):
            poolv = F.adaptive_avg_pool2d(v_l, (round(H / pool_ratio), round(W / pool_ratio)))
            poolv = poolv + l(poolv)  # 论文中说是相对位置编码  利用卷积
            # fix backward bug in higher torch versions when training
            poolvs_l.append(poolv.view(B, C, -1))  #####原
            # pools.append(pool)  #####后


        ks = torch.cat(pools_h, dim=2)  #torch.Size([3, 64, 54])原尺寸56，56-ratio[12,16,20,24]
        ks = self.norm(ks.permute(0, 2, 1))  #####原 torch.Size([3, 54, 64])
        vs = torch.cat(poolvs_h, dim=2)
        vs = self.norm(vs.permute(0, 2, 1)) # torch.Size([3, 64, 54])
        # pools = self.norm(pools) #####后
        ksl = torch.cat(pools_l, dim=2)  # torch.Size([3, 64, 54])原尺寸56，56-ratio[12,16,20,24]
        ksl = self.hlnorm(ksl.permute(0, 2, 1))  #####原 torch.Size([3, 191, 64])
        vsl = torch.cat(poolvs_l, dim=2)
        vsl = self.hlnorm(vsl.permute(0, 2, 1))


        k = ks.reshape(B, -1, self.num_heads, Cl // self.num_heads).permute( 0, 2, 1, 3)  #torch.Size([3, 2, 54, 32])尺寸56，56
        # kk----------torch.Size([3, 2, 191, 32])
        v = vs.reshape(B, -1, self.num_heads, Cl // self.num_heads).permute(0, 2, 1, 3)
        # v----torch.Size([3, 2, 191, 32])

        kl = ksl.reshape(B, -1, self.num_heads, Cl // self.num_heads).permute(0, 2, 1,
                                                                           3)  # torch.Size([3, 2, 54, 32])尺寸56，56
        # kk----------torch.Size([3, 2, 191, 32])
        vl = vsl.reshape(B, -1, self.num_heads, Cl // self.num_heads).permute(0, 2, 1, 3)

        # q_h, v_h = map(
        #     lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=C // self.num_heads,
        #                         heads=self.num_heads, h=H, w=W), [q_h, v_h])  # torch.Size([3, 2, 3136, 32])
        # q_l, v_l = map(
        #     lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=C // self.num_heads,
        #                         heads=self.num_heads, h=Hl, w=Wl), [q_l, v_l])  # torch.Size([3, 2, 3136, 32])


         #  q----torch.Size([3, 2, 12544, 32]) 112x112


        # 矩阵乘法
        # q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        #  attn-----torch.Size([3, 2, 12544, 191])   v----torch.Size([3, 2, 191, 32])
        attn = (q_gl @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  #torch.Size([3, 2, 3137, 54])

        #  attn-----torch.Size([3, 2, 12544, 191])   v----torch.Size([3, 2, 191, 32])
        attnl = (q_gh @ kl.transpose(-2, -1)) * self.scale
        attnl = attnl.softmax(dim=-1)
        attnl = self.attn_drop(attnl)
        out_attl = (attnl @ vl)  ###high q  low k,v----得到low调整的high # torch.Size([3, 2, 3137, 32])

        out_att = (attn @ v)  ###low q  high k,v  得到high调整的low  # torch.Size([3, 2, 3137, 32])原尺寸56，56
        # torch.Size([3, 2, 12544, 32])
        # x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        glob_token, att_fea = out_att[:,:,0],out_att[:,:,1:] # torch.Size([3, 2, 32])，torch.Size([3, 2, 3136, 32])

        glob_tokens = rearrange(glob_token, 'b h d -> b h d 1') #torch.Size([3, 2, 32，1]

        glob_tokenl, att_feal = out_attl[:, :, 0], out_attl[:, :, 1:]  # torch.Size([3, 2, 32])，torch.Size([3, 2, 3136, 32])

        glob_tokensl = rearrange(glob_tokenl, 'b h d -> b h d 1')

        # w_ql w_kl----torch.Size([3, 2, 32, 1])
        w_ql, w_kl = self.global_tokens_to_qk(glob_tokensl).chunk(2,
                                                               dim=2)  # dim--1torch.Size([3, 1, 64, 1]) dim--2torch.Size([3, 2, 32, 1])
        w_ql = w_ql * self.gloscale

        # similarities
        # torch.Size([3, 2, 32, 32])
        w_dotsl = einsum('b h i d, b h j d -> b h i j', w_ql, w_kl)
        # torch.Size([3, 2, 32, 32])
        w_attnl = self.global_attend(w_dotsl)  # dim--1torch.Size([3, 2, 32, 32])dim-2torch.Size([3, 1, 64, 64])
        # att_fea----torch.Size([3, 2, 3136, 32])
        # torch.Size([3, 2, 3136, 32])
        aggregated_glo_fmapl = einsum('b h i j, b h j w -> b h i w', att_feal, w_attnl)
        # 得到low调整的high torch.Size([3, 64, 56, 56])
        outl = rearrange(aggregated_glo_fmapl, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W,
                        dim_head=C // self.num_heads,
                        heads=self.num_heads)


        w_q, w_k = self.hlglobal_tokens_to_qk(glob_tokens).chunk(2, dim=2)  # dim--1torch.Size([3, 1, 64, 1]) dim--2torch.Size([3, 2, 32, 1])
        w_q = w_q * self.gloscale

        # similarities
        # torch.Size([24, 8, 64, 64])
        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)
        w_attn = self.global_attend(w_dots)  #dim--1torch.Size([3, 2, 32, 32])dim-2torch.Size([3, 1, 64, 64])
        aggregated_glo_fmap = einsum('b h i j, b h j w -> b h i w', att_fea, w_attn)
        # torch.Size([3, 2, 3136, 32])
        # 得到high调整的low
        out = rearrange(aggregated_glo_fmap, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=C // self.num_heads,
                        heads=self.num_heads)
        # torch.Size([3, 64, 56, 56])

        prj_outl = self.out_l(out)
        outl = self.proj_drop(prj_outl)  # torch.Size([3, 64, 56, 56])

        prj_outh = self.out_h(outl)
        outh = self.proj_drop(prj_outh) # torch.Size([3, 64, 56, 56])

        return outh,outl

class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[1, 2, 3, 6]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        # self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        # self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        #########原先###################
        # self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        # self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        ###########后来#############
        self.q = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, padding=0, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim) ##原----
        # self.norm = nn.BatchNorm2d(dim)  # 后

    def forward(self, x, d_convs=None):
        # B, N, C = x.shape  # [3, 3136, 64]
        B, C, H, W = x.shape
        # B, inner_dim, H, W
        # qkv = self.to_qkv(x)
        # q, k, v = qkv.chunk(3, dim=1)
        q = self.q(x)
        # torch.Size([3, 2, 3136, 32])
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        xpoo = self.kv(x)
        k, v = xpoo.chunk(2, dim=1)
        pools = []
        poolvs = []

        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            pool = F.adaptivep_avg_pool2d(k, (round(H / pool_ratio), round(W / pool_ratio)))
            pool = pool + l(pool)  # 论文中说是相对位置编码  利用卷积
            # fix backward bug in higher torch versions when training
            pools.append(pool.view(B, C, -1))  #####原
            # pools.append(pool)  #####后
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            poolv = F.adaptive_avg_pool2d(v, (round(H / pool_ratio), round(W / pool_ratio)))
            poolv = poolv + l(poolv)  # 论文中说是相对位置编码  利用卷积
            # fix backward bug in higher torch versions when training
            poolvs.append(poolv.view(B, C, -1))  #####原
            # pools.append(pool)  #####后


        ks = torch.cat(pools, dim=2)
        ks = self.norm(ks.permute(0, 2, 1))  #####原 torch.Size([3, 191, 64])
        vs = torch.cat(poolvs, dim=2)
        vs = self.norm(vs.permute(0, 2, 1))
        # pools = self.norm(pools) #####后


        k = ks.reshape(B, -1, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
        # kk----------torch.Size([3, 2, 191, 32])
        v = vs.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # v----torch.Size([3, 2, 191, 32])



         #  q----torch.Size([3, 2, 12544, 32]) 112x112
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=C // self.num_heads, heads=self.num_heads,
                      h=H, w=W)

        # 矩阵乘法
        # q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        #  attn-----torch.Size([3, 2, 12544, 191])   v----torch.Size([3, 2, 191, 32])
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out_att = (attn @ v)  # torch.Size([3, 2, 12544, 32])
        # x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        out = rearrange(out_att, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=C // self.num_heads,
                        heads=self.num_heads)

        prj_out = self.proj(out)
        out = self.proj_drop(prj_out)

        return out
class Glo_tkPoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[1, 2, 3, 6]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        # self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        # self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        #########原先###################
        # self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        # self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        ###########后来#############
        self.q = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, padding=0, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim) ##原----
        # self.norm = nn.BatchNorm2d(dim)  # 后
        # window tokens

        self.global_tokens = nn.Parameter(torch.randn(dim))

        # prenorm and non-linearity for window tokens
        # then projection to queries and keys for window tokens

        self.global_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(1), #补充的global——token为[3,64,1]
            nn.GELU(),# [3,2,32,1]
            Rearrange('b h n c -> b (h n) c'),
            nn.Conv1d(dim, dim * 2, 1),
            Rearrange('b (h n) c -> b h n c', h=num_heads),
        )

        # window attention

        self.global_attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(proj_drop)
        )

        # self.to_out = nn.Sequential(
        #     nn.Conv2d(dim, dim, 1),
        #     nn.Dropout(proj_drop)
        # )

    def forward(self, x, d_convs=None):
        # B, N, C = x.shape  # [3, 3136, 64]
        B, C, H, W = x.shape
        # B, inner_dim, H, W
        # qkv = self.to_qkv(x)
        # q, k, v = qkv.chunk(3, dim=1)

        q = self.q(x)
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=C // self.num_heads,
                      heads=self.num_heads,
                      h=H, w=W)  # torch.Size([3, 2, 3136, 32])
        # q1 = q.transpose(-2,-1)
        glo = repeat(self.global_tokens, 'c-> b c 1 ', b=B)  # torch.Size([3, 64, 1])
        glo = rearrange(glo, 'b (dim_head heads) 1 -> b heads 1 dim_head', dim_head=C // self.num_heads,
                        heads=self.num_heads,
                        )  #torch.Size([3, 2, 1, 32])
        # glo1 = glo.transpose(-2,-1)  #torch.Size([3, 2, 32, 1])
        q_g = torch.cat((glo, q), dim=2)  # torch.Size([3, 2, 3137, 32])
        # torch.Size([3, 2, 3136, 32])
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        xpoo = self.kv(x) # torch.Size([3, 128, 56, 56])
        k, v = xpoo.chunk(2, dim=1)  #torch.Size([3, 64, 56, 56])
        pools = []
        poolvs = []

        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            pool = F.adaptive_avg_pool2d(k, (round(H / pool_ratio), round(W / pool_ratio)))
            #原尺寸56--ratio--12--torch.Size([3, 64, 5, 5]) ratio--16--torch.Size([3, 64, 4, 4])
            # ratio--20--torch.Size([3, 64, 3, 3]) ratio--24--torch.Size([3, 64, 2, 2])
            pool = pool + l(pool)  # 论文中说是相对位置编码  利用卷积
            # fix backward bug in higher torch versions when training
            pools.append(pool.view(B, C, -1))  #####原
            # pools.append(pool)  #####后
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            poolv = F.adaptive_avg_pool2d(v, (round(H / pool_ratio), round(W / pool_ratio)))
            poolv = poolv + l(poolv)  # 论文中说是相对位置编码  利用卷积
            # fix backward bug in higher torch versions when training
            poolvs.append(poolv.view(B, C, -1))  #####原
            # pools.append(pool)  #####后


        ks = torch.cat(pools, dim=2)  #torch.Size([3, 64, 54])原尺寸56，56-ratio[12,16,20,24]
        ks = self.norm(ks.permute(0, 2, 1))  #####原 torch.Size([3, 191, 64])
        vs = torch.cat(poolvs, dim=2)
        vs = self.norm(vs.permute(0, 2, 1))
        # pools = self.norm(pools) #####后


        k = ks.reshape(B, -1, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)  #torch.Size([3, 2, 54, 32])尺寸56，56
        # kk----------torch.Size([3, 2, 191, 32])
        v = vs.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # v----torch.Size([3, 2, 191, 32])



         #  q----torch.Size([3, 2, 12544, 32]) 112x112


        # 矩阵乘法
        # q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        #  attn-----torch.Size([3, 2, 12544, 191])   v----torch.Size([3, 2, 191, 32])
        attn = (q_g @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  #torch.Size([3, 2, 3137, 54])

        out_att = (attn @ v)  # torch.Size([3, 2, 3137, 32])原尺寸56，56
        # torch.Size([3, 2, 12544, 32])
        # x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        glob_token, att_fea = out_att[:,:,0],out_att[:,:,1:] # torch.Size([3, 2, 32])，torch.Size([3, 2, 3136, 32])

        glob_tokens = rearrange(glob_token, 'b h d -> b h d 1') #torch.Size([3, 2, 32，1]
        # windowed_fmaps = rearrange(windowed_fmaps, '(b x y) h n d -> b h (x y) n d', x=height // wsz, y=width // wsz)

        # windowed queries and keys (preceded by prenorm activation)
        # torch.Size([24, 8, 64, 32])
        #glo_#torch.Size([3, 2, 32，1]
        w=self.global_tokens_to_qk(glob_tokens)  #torch.Size([3, 2, 64, 1])
        w_q, w_k = self.global_tokens_to_qk(glob_tokens).chunk(2, dim=2)  # dim--1torch.Size([3, 1, 64, 1]) dim--2torch.Size([3, 2, 32, 1])
        w_q = w_q * self.scale

        # similarities
        # torch.Size([24, 8, 64, 64])
        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)
        # torch.Size([24, 8, 64, 64])
        w_attn = self.global_attend(w_dots)  #dim--1torch.Size([3, 2, 32, 32])dim-2torch.Size([3, 1, 64, 64])
        # att_fea----torch.Size([3, 2, 3136, 32])
        # att_fea.transpose(-2,-1)
        aggregated_glo_fmap = einsum('b h i j, b h j w -> b h i w', att_fea, w_attn)
        # torch.Size([3, 2, 3136, 32])

        out = rearrange(aggregated_glo_fmap, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=C // self.num_heads,
                        heads=self.num_heads)
        # torch.Size([3, 64, 56, 56])

        prj_out = self.proj(out)
        out = self.proj_drop(prj_out)

        return out


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, pool_ratios=[12, 16, 20, 24]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, pool_ratios=pool_ratios)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=drop,
                       ksize=3)

    def forward(self, x, d_convs=None):
        x = self.norm1(x)
        x = x + self.drop_path(self.attn(x, d_convs=d_convs))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
class Glo_tkBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, pool_ratios=[12, 16, 20, 24]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Glo_tkPoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, pool_ratios=pool_ratios)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=drop,
                       ksize=3)

    def forward(self, x, d_convs=None):
        x = self.norm1(x)
        x = x + self.drop_path(self.attn(x, d_convs=d_convs))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ (Overlapped) Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=3, in_chans=3, embed_dim=768, overlap=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        if not overlap:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                                  padding=kernel_size // 2)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # stage2--torch.Size([8, 96, 28, 28])
        # stage2--proj---Conv2d(48, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # torch.Size([8, 784, 96])  56->28  28x28=784
        x = self.norm(x)

        return x


#########################################conv ###############
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

#

class down_block_pv(nn.Module):
    def __init__(self, in_ch, bottleneck=False, maxpool=True, num_croblock=0, embed_dims=0,depths=2,dpr=[],cur=0,
                 heads=4, attn_drop=0.0, drop_rate=0., mlp_ratios=8, qkv_bias=True, qk_scale=None, pool_ratios=20,):

        super().__init__()

        block_list = []
        self.Attblock = nn.ModuleList([Block(
            dim=embed_dims, num_heads=heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop, drop_path=dpr[cur + i], norm_layer=nn.BatchNorm2d,
            pool_ratios=pool_ratios)
            for i in range(depths)])
        self.d_convs1 = nn.ModuleList(
            [nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, groups=embed_dims) for temp
             in pool_ratios])


        if bottleneck:
            block = Bottleneck
            # block =
        else:
            block = BasicBlock

        if maxpool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, embed_dims, stride=1))
        else:
            block_list.append(block(in_ch, embed_dims, stride=2))


        # for i in range(num_croblock):
        #     block_list.append(block(out_ch, out_ch, stride=1))
        #     block_list.append(Attblock(imgsize=imgsize))

        self.blocks = nn.Sequential(*block_list)

    def forward(self, x):

        out_conv = self.blocks(x)
        B,_,H,W = out_conv.shape
        for idx, blk in enumerate(self.Attblock):
            out = blk(out_conv,self.d_convs1)
        # out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out
class Glo_tkdown_block_pv(nn.Module):
    def __init__(self, in_ch, bottleneck=False, maxpool=True, num_croblock=0, embed_dims=0,depths=2,dpr=[],cur=0,
                 heads=4, attn_drop=0.0, drop_rate=0., mlp_ratios=8, qkv_bias=True, qk_scale=None, pool_ratios=20,):

        super().__init__()

        block_list = []
        self.Attblock = nn.ModuleList([Glo_tkBlock(
            dim=embed_dims, num_heads=heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop, drop_path=dpr[cur + i], norm_layer=nn.BatchNorm2d,
            pool_ratios=pool_ratios)
            for i in range(depths)])
        self.d_convs1 = nn.ModuleList(
            [nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, groups=embed_dims) for temp
             in pool_ratios])


        if bottleneck:
            block = Bottleneck
            # block =
        else:
            block = BasicBlock

        if maxpool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, embed_dims, stride=1))
        else:
            block_list.append(block(in_ch, embed_dims, stride=2))


        # for i in range(num_croblock):
        #     block_list.append(block(out_ch, out_ch, stride=1))
        #     block_list.append(Attblock(imgsize=imgsize))

        self.blocks = nn.Sequential(*block_list)

    def forward(self, x):

        out_conv = self.blocks(x)
        B,_,H,W = out_conv.shape
        for idx, blk in enumerate(self.Attblock):
            out = blk(out_conv,self.d_convs1)
        # out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out

class up_block_pv(nn.Module):
    def __init__(self, in_ch, bottleneck=False, interplot=True, num_croblock=0, embed_dims=0,depths=2,dpr=[],cur=0,
                 heads=4, attn_drop=0.0, drop_rate=0., mlp_ratios=8, qkv_bias=True, qk_scale=None, pool_ratios=20,):

        super().__init__()
        self.interplot = interplot

            # self.conv_ch = nn.Conv2d(in_ch, embed_dims, kernel_size=1)
        # self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch,
        #                            stride=1)


        block_list = []
        self.Attblock = nn.ModuleList([Block(
            dim=embed_dims, num_heads=heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop, drop_path=dpr[cur + i], norm_layer=nn.BatchNorm2d,
            pool_ratios=pool_ratios)
            for i in range(depths)])
        self.d_convs1 = nn.ModuleList(
            [nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, groups=embed_dims) for temp
             in pool_ratios])


        if bottleneck:
            block = Bottleneck
            # block =
        else:
            block = BasicBlock

        if interplot:
            self.conv_ch = depthwise_separable_conv(in_ch, embed_dims)
            block_list.append(block(2*embed_dims, embed_dims, stride=1))
        else:
            self.upconv = nn.ConvTranspose2d(in_ch, embed_dims, kernel_size=2, stride=2)
            block_list.append(block(2*embed_dims, embed_dims, stride=1))


        # for i in range(num_croblock):
        #     block_list.append(block(out_ch, out_ch, stride=1))
        #     block_list.append(Attblock(imgsize=imgsize))

        self.blocks = nn.Sequential(*block_list)

    def forward(self, x1,x2):

        if self.interplot:
           x_o = F.interpolate(self.conv_ch(x1), size=x2.shape[-2:], mode='bilinear', align_corners=True)
        else:
           x_o = self.upconv(x1)
        out = torch.cat([x_o, x2], dim=1)
        out= self.blocks(out)
        B,_,H,W = out.shape

        if H != 224:
          for idx, blk in enumerate(self.Attblock):
            out = blk(out,self.d_convs1)
        # out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out


if __name__ == '__main__':
    # net = ConvPv(3, 32, 9, num_heads=[2,4,5,8], num_croblock=[0, 0, 0, 0],
    #                    attn_drop=0.1, maxpool=True)
    # print(net)
    pool_ratios=[12, 16, 20, 24]
    d_convs1 = nn.ModuleList(
        [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64) for temp
         in pool_ratios])
    attn = Glo_tkPoolingAttention(
        64, num_heads=2, qkv_bias=True, qk_scale=None,
        attn_drop=0., proj_drop=0., pool_ratios=[12, 16, 20, 24])
    tx = Glo_skipAttention(64, num_heads=2, qkv_bias=True, qk_scale=None,
        attn_drop=0., proj_drop=0., pool_ratios=[12, 16, 20, 24])

    x = torch.rand(3, 64, 56, 56)
    y = torch.rand(3, 64, 56, 56)
    out1,out2 = tx(x,y,d_convs1)
    print(out1.shape,out2.shape)
    # net.to("cuda")
    # prewe = torch.load("./deTup_pre/147_sc_0__best.pth",map_location=torch.device('cpu'))
    # net.load_state_dict(prewe, strict=False)

    input = torch.rand(3, 64, 56, 56)
    outp = attn(input,d_convs1)
    print(outp.size())



