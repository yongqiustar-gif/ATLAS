# atlas/nn_layers.py
from __future__ import annotations
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

from ._gpublock import AtlasGPULayer


class TimeEncode(nn.Module):
    def __init__(self, dim_time: int):
        super().__init__()
        self.dim_time = dim_time
        self.w = nn.Linear(1, dim_time)
        self.w.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time))
            .float().view(dim_time, 1)
        )
        self.w.bias = nn.Parameter(torch.zeros(dim_time))
    def forward(self, ts: Tensor) -> Tensor:
        return torch.cos(self.w(ts.unsqueeze(-1)))


class AtlasTemporalAttnLayer(nn.Module):
    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_time: int,
        dim_out: int,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim_out % num_heads == 0
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_time = dim_time
        self.dim_out  = dim_out
        self.num_heads = num_heads

        self.time_enc = TimeEncode(dim_time)

        self.w_q  = nn.Linear(dim_node + dim_time, dim_out)
        self.w_kv = nn.Linear(dim_node + dim_edge + dim_time, dim_out * 2)
        self.w_out = nn.Linear(dim_node + dim_out, dim_out)

        self.attn_act = nn.LeakyReLU(0.2)
        self.dropout  = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_out)

    def forward(self, layer: AtlasGPULayer) -> Tensor:
        dst_feat = layer.dst_feat
        src_feat = layer.src_feat
        edge_feat = layer.edge_feat
        dst_ts = layer.dst_ts
        src_ts = layer.src_ts

        N_l, K = src_feat.shape[0], src_feat.shape[1]
        Dn = dst_feat.shape[1]
        De = 0 if edge_feat is None else edge_feat.shape[2]

        delta_t = (dst_ts.unsqueeze(-1) - src_ts).clamp(min=0.0)
        t_dst = self.time_enc(dst_ts)
        t_src = self.time_enc(delta_t.reshape(-1)).reshape(N_l, K, -1)

        Q = self.w_q(torch.cat([dst_feat, t_dst], dim=-1))

        if De > 0:
            nbr_in = torch.cat([src_feat, edge_feat, t_src], dim=-1)
        else:
            nbr_in = torch.cat([src_feat, t_src], dim=-1)

        K_mat, V_mat = self.w_kv(nbr_in).chunk(2, dim=-1)

        H = self.num_heads
        dh = self.dim_out // H
        Qh = Q.view(N_l, H, dh)
        Kh = K_mat.view(N_l, K, H, dh).transpose(1, 2)
        Vh = V_mat.view(N_l, K, H, dh).transpose(1, 2)

        attn = torch.einsum("nhd,nhkd->nhk", Qh, Kh)
        attn = self.dropout(torch.softmax(self.attn_act(attn), dim=-1))

        out = torch.einsum("nhk,nhkd->nhd", attn, Vh).reshape(N_l, self.dim_out)
        out = torch.cat([out, dst_feat], dim=-1)
        out = self.w_out(out)
        out = F.relu(self.dropout(out))
        out = self.layer_norm(out)
        return out



class AtlasTemporalMemoryAttnLayer(nn.Module):
    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_time: int,
        dim_out: int,
        dim_embed: int,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim_out % num_heads == 0
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_time = dim_time
        self.dim_embed = dim_embed
        self.dim_out  = dim_out
        self.num_heads = num_heads

        self.mem2node = nn.Linear(dim_embed, dim_node)
        self.time_enc = TimeEncode(dim_time)

        self.w_q  = nn.Linear(dim_node + dim_time, dim_out)
        self.w_kv = nn.Linear(dim_node + dim_edge + dim_time, dim_out * 2)
        self.w_out = nn.Linear(dim_node + dim_out, dim_out)

        self.attn_act = nn.LeakyReLU(0.2)
        self.dropout  = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_out)

    def forward(self, layer: AtlasGPULayer, memory: torch.tensor) -> Tensor:

        dst_nodes, src_nodes = layer.dst_nodes, layer.src_nodes

        dst_feat = layer.dst_feat + self.mem2node(memory[dst_nodes])
        src_feat = layer.src_feat + self.mem2node(memory[src_nodes])
        edge_feat = layer.edge_feat
        dst_ts = layer.dst_ts
        src_ts = layer.src_ts

        N_l, K = src_feat.shape[0], src_feat.shape[1]
        Dn = dst_feat.shape[1]
        De = 0 if edge_feat is None else edge_feat.shape[2]

        delta_t = (dst_ts.unsqueeze(-1) - src_ts).clamp(min=0.0)
        t_dst = self.time_enc(dst_ts)
        t_src = self.time_enc(delta_t.reshape(-1)).reshape(N_l, K, -1)

        Q = self.w_q(torch.cat([dst_feat, t_dst], dim=-1))

        if De > 0:
            nbr_in = torch.cat([src_feat, edge_feat, t_src], dim=-1)
        else:
            nbr_in = torch.cat([src_feat, t_src], dim=-1)

        K_mat, V_mat = self.w_kv(nbr_in).chunk(2, dim=-1)

        H = self.num_heads
        dh = self.dim_out // H
        Qh = Q.view(N_l, H, dh)
        Kh = K_mat.view(N_l, K, H, dh).transpose(1, 2)
        Vh = V_mat.view(N_l, K, H, dh).transpose(1, 2)

        attn = torch.einsum("nhd,nhkd->nhk", Qh, Kh)
        attn = self.dropout(torch.softmax(self.attn_act(attn), dim=-1))

        out = torch.einsum("nhk,nhkd->nhd", attn, Vh).reshape(N_l, self.dim_out)
        out = torch.cat([out, dst_feat], dim=-1)
        out = self.w_out(out)
        out = F.relu(self.dropout(out))
        out = self.layer_norm(out)
        return out
