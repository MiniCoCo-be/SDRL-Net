import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import pdb
from torch import nn, einsum


# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

def create_grid_like(t, dim = 0):
    h, w, device = *t.shape[-2:], t.device
    # torch.meshgrid()
    grid = torch.stack(torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device)), dim=dim)
    # grid = torch.stack(torch.meshgrid(
    #     torch.arange(h, device = device),
    #     torch.arange(w, device = device),
    # indexing = 'ij'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim = dim)

    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_h, grid_w), dim = out_dim)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# continuous positional bias from SwinV2
class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype
# grid_kv torch.Size([8, 16, 16, 2]) grid_q torch.Size([64, 64, 2])
        grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')  # torch.Size([1, 4096, 2])
        grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c') # torch.Size([8, 256, 2])

        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')  # torch.Size([8, 4096, 256, 2])
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # # torch.Size([8, 4096, 256, 2])
        # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

        return bias

# main class

class DeformDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_dim,
        out_dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        downsample_factor = 4,
        offset_scale = None,
        offset_groups = None,
        offset_kernel_size = 6,
        group_queries = True,
        group_key_values = True
    ):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor

        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.rel_pos_bias = CPB(out_dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv2d(out_dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
        self.to_qh = nn.Conv2d(in_dim, inner_dim, 1, groups=offset_groups if group_queries else 1, bias=False)
        self.to_k = nn.Conv2d(in_dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_v = nn.Conv2d(in_dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, out_dim, 1)

    def forward(self,qh, x, return_vgrid = False):
        """
        qh当作query,x当作key、value
        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        g - offset groups
        """

        heads, b, h, w, downsample_factor, device = self.heads, qh.shape[0], *qh.shape[-2:], self.downsample_factor, x.device

        # queries

        q = self.to_q(qh)
        qhiof = self.to_qh(x)

        # calculate offsets - offset MLP shared across all groups
        # x---[1,512,64,64]
        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)
        # 分组group
        # torch.Size([8, 16, 32, 32])
        grouped_queries = group(qhiof)  # torch.Size([8, 64, 64, 64])
        offsets = self.to_offsets(grouped_queries) # torch.Size([8, 2, 16, 16])

        # calculate grid + offsets

        grid =create_grid_like(offsets) # torch.Size([2, 16, 16])
        vgrid = grid + offsets  # torch.Size([8, 2, 16, 16])

        vgrid_scaled = normalize_grid(vgrid)

        kv_feats = F.grid_sample(   # torch.Size([8, 64, 16, 16])
            group(x),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
       # torch.Size([8, 64, 16, 16])
        # torch.Size([1, 512, 16, 16])
        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)

        # derive key / values

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)
        # torch.Size([1, 128, 8, 8])
        # scale queries

        q = q * self.scale

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # query / key similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # relative positional bias

        grid = create_grid_like(qh) # torch.Size([2, 64, 64])
        grid_scaled = normalize_grid(grid, dim = 0)
        rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias
        # sim torch.Size([1, 8, 4096, 64]) relpo torch.Size([1, 8, 4096, 64])
        # numerical stability
        s= sim.amax(dim = -1, keepdim = True)  # torch.Size([1, 8, 4096, 1])
        o = sim.amax(dim = -1, keepdim = True).detach()  # torch.Size([1, 8, 4096, 1])

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate and combine heads
         # attn torch.Size([1, 8, 4096, 64])  v torch.Size([1, 8, 64, 16])
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        return out



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
    def __init__(self, in_ch, hid_ch=None, out_ch=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_ch = out_ch or in_ch
        hid_ch = hid_ch or in_ch

        self.fc1 = nn.Conv2d(in_ch, hid_ch, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hid_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

# class DeformAttentionDecoder(nn.Module):
#
#     def __init__(self, in_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16,
#                  projection='interp', rel_pos=True):
#         super().__init__()
#
#         self.inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head ** (-0.5)
#         self.dim_head = dim_head
#         self.reduce_size = reduce_size
#         self.projection = projection
#         self.rel_pos = rel_pos
#
#         # depthwise conv is slightly better than conv1x1
#         # self.to_kv = nn.Conv2d(dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=True)
#         # self.to_q = nn.Conv2d(dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
#
#         self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim * 2)
#         self.to_q = depthwise_separable_conv(out_dim, self.inner_dim)
#         self.to_out = depthwise_separable_conv(self.inner_dim, out_dim)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         if self.rel_pos:
#             self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
#             # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)
#
#     def forward(self, q, x):
#
#         B, C, H, W = x.shape  # low-res feature shape
#         BH, CH, HH, WH = q.shape  # high-res feature shape
#
#         # low-res feature 求k,v 最后对低特征进行特征加重
#         k, v = self.to_kv(x).chunk(2, dim=1)  # B, inner_dim, H, W
#         q = self.to_q(q)  # BH, inner_dim, HH, WH
#
#         if self.projection == 'interp' and H != self.reduce_size:
#             k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))
#
#         elif self.projection == 'maxpool' and H != self.reduce_size:
#             k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
#
#         q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
#                       h=HH, w=WH)
#         k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
#                                        heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
#
#         q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
#
#         if self.rel_pos:
#             relative_position_bias = self.relative_position_encoding(HH, WH)
#             q_k_attn += relative_position_bias
#             # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, HH, WH, self.dim_head)
#             # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w
#
#         q_k_attn *= self.scale
#         q_k_attn = F.softmax(q_k_attn, dim=-1)
#         q_k_attn = self.attn_drop(q_k_attn)
#
#         out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
#         out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head,
#                         heads=self.heads)
#
#         out = self.to_out(out)
#         out = self.proj_drop(out)
#
#         return out, q_k_attn


class DeformDecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, heads, attn_drop=0.):
        super().__init__()

        self.bn_l = nn.BatchNorm2d(in_ch)
        self.bn_h = nn.BatchNorm2d(out_ch)

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.attn = DeformDecoder(in_ch, out_ch, heads=heads, dim_head=out_ch//heads, dropout=attn_drop, downsample_factor=4,offset_scale=4)

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        # 低特征上采样
        # residue = F.interpolate(self.conv_ch(x1), size=x2.shape[-2:], mode='bilinear', align_corners=True)
        # x1: low-res, x2: high-res
        x1 = self.bn_l(x1)
        x2 = self.bn_h(x2)
        # 主要使用低分辨率特征
        out, q_k_attn = self.attn(x2, x1)

        # # out = out + residue
        # residue = out
        #
        # out = self.bn2(out)
        # out = self.relu(out)
        # out = self.mlp(out)

        # out += residue

        return out

if __name__ == '__main__':
    attn = DeformDecoder(
        in_dim=256,
        out_dim=128,# feature dimensions
        dim_head=16,  # dimension per head
        heads=8,  # attention heads
        dropout=0.,  # dropout
        downsample_factor=4,  # downsample factor (r in paper)
        offset_scale=4,  # scale of offset, maximum offset
        offset_groups=None,  # number of offset groups, should be multiple of heads
        offset_kernel_size=6,  # offset kernel size
    )
    q = torch.randn(1, 128, 64, 64)
    x = torch.randn(1, 256, 32, 32)
    out= attn(q,x)  # (1, 512, 64, 64)
    print(out.shape,"done---------------------------")

