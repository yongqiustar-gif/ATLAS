# atlas_jodie.py
from __future__ import annotations
from typing import Optional


import math
import os
import sys

import torch
from torch import nn, Tensor

import atlas as at
from atlas import AtlasGPUBlock, AtlasGPULayer

# Add the examples directory (where support.py is located) to sys.path
THIS_DIR = os.path.dirname(__file__)
EXAMPLES_DIR = os.path.dirname(THIS_DIR)
if EXAMPLES_DIR not in sys.path:
    sys.path.append(EXAMPLES_DIR)

import support


class NormalLinear(nn.Linear):
    def reset_parameters(self) -> None:  # type: ignore[override]
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.normal_(self.weight, mean=0.0, std=stdv)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=stdv)


class AtlasJodie(nn.Module):
    def __init__(
        self, 
        g: at.TGraph,
        dim_node: int,
        dim_edge: int,
        dim_time: int,
        dim_embed: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.g = g
        self.dim_edge = dim_edge
        self.dim_node = dim_node
        self.dim_embed = dim_embed
        self.dim_time = dim_time
        self.nfeat_map = None if dim_node == dim_embed else nn.Linear(dim_node, dim_embed)
        self.mail_dim = dim_embed + dim_edge

        mem_init = torch.zeros(g.num_nodes(), dim_embed, dtype=torch.float32, device=device)
        ts_init = torch.zeros(g.num_nodes(), dtype=torch.float32, device=device)
        self.register_buffer("memory", mem_init)
        self.register_buffer("memory_ts", ts_init)
        mail_init = torch.zeros(g.num_nodes(), self.mail_dim, dtype=torch.float32, device=device)
        mail_ts_init = torch.zeros(g.num_nodes(), dtype=torch.float32, device=device)
        self.register_buffer("mail", mail_init)
        self.register_buffer("mail_ts", mail_ts_init)

        dim_input = dim_embed + dim_edge + dim_time
        self.updater = nn.RNNCell(dim_input, dim_embed)
        self.time_encode = at.nn_layers.TimeEncode(dim_time)
        self.time_linear = NormalLinear(1, dim_embed)
        self.norm = nn.LayerNorm(dim_embed)
        self.edge_predictor = support.EdgePredictor(dim_embed)
        

    def forward(self, block: AtlasGPUBlock) -> tuple[Tensor, Tensor]:
        B = block.batch_size
        if B == 0:
            raise RuntimeError("AtlasJodie received empty block")

        layer0 = block.layers[0]
        root_nodes = layer0.dst_nodes.clone()

        prev_mem = self.memory[root_nodes].detach()
        prev_ts = self.memory_ts[root_nodes].detach()
        mail = self.mail[root_nodes].detach()
        mail_ts = self.mail_ts[root_nodes].detach()
        delta_t = mail_ts - prev_ts
        time_feat = self.time_encode(delta_t)

        updater_in = torch.cat([mail, time_feat], dim=1)
        updated = self.updater(updater_in, prev_mem)

        normalized = self._normalize_embed(layer0, updated)

        pos_nodes = layer0.dst_nodes[: 2 * B].clone()
        self.memory[pos_nodes] = normalized[: 2 * B].detach()
        self.memory_ts[pos_nodes] = mail_ts[: 2 * B].detach()
        self._store_mail(block, normalized)

        projected = self._project_embed(normalized, mail_ts, block.root_ts)

        src_h = projected[0:B]
        dst_h = projected[B : 2 * B]
        neg_h = projected[2 * B : 3 * B]

        pos_scores = self.edge_predictor(src_h, dst_h)
        neg_scores = self.edge_predictor(src_h, neg_h)
        return pos_scores, neg_scores

    def _normalize_embed(self, layer0: AtlasGPULayer, embed: Tensor) -> Tensor:
        if layer0.dst_feat is None:
            return self.norm(embed)

        feats = layer0.dst_feat.to(embed.device)
        if feats.shape[-1] == self.dim_embed:
            mapped = feats
        elif feats.shape[-1] == self.dim_node and self.nfeat_map is not None:
            mapped = self._map_feat(feats)
        elif feats.shape[-1] > self.dim_embed:
            mapped = feats[..., : self.dim_embed]
        else:
            pad = torch.zeros(*feats.shape[:-1], self.dim_embed - feats.shape[-1], device=embed.device, dtype=feats.dtype)
            mapped = torch.cat([feats, pad], dim=-1)
        embed = embed + mapped
        return self.norm(embed)

    def _project_embed(self, embed: Tensor, mail_ts: Tensor, edge_ts: Tensor) -> Tensor:
        times = edge_ts.repeat(3).to(embed.device)
        delta = times - mail_ts
        denom = (times + 1.0).unsqueeze(-1)
        time_diff = (delta.unsqueeze(-1) / denom).to(embed.device)
        return embed * (1 + self.time_linear(time_diff))

    def _store_mail(self, block: AtlasGPUBlock, normalized: Tensor) -> None:
        B = block.batch_size
        if B == 0:
            return

        layer0 = block.layers[0]
        pos_nodes = layer0.dst_nodes[: 2 * B].clone()
        device = normalized.device

        mail_embed = torch.empty(2 * B, self.dim_embed, device=device)
        mail_embed[:B] = normalized[B : 2 * B]
        mail_embed[B : 2 * B] = normalized[0:B]

        if self.dim_edge > 0:
            if block.root_edge_feat is not None:
                edge_feat = block.root_edge_feat.to(device)
            else:
                edge_feat = torch.zeros(B, self.dim_edge, device=device)
            edge_pairs = torch.cat([edge_feat, edge_feat], dim=0)
            mail = torch.cat([mail_embed, edge_pairs], dim=1)
        else:
            mail = mail_embed

        self.mail[pos_nodes] = mail.detach()
        ts = block.root_ts.repeat(2).to(device)
        self.mail_ts[pos_nodes] = ts.detach()

    def _map_feat(self, feat: Tensor) -> Tensor:
        assert self.nfeat_map is not None
        orig_shape = feat.shape
        mapped = self.nfeat_map(feat.reshape(-1, self.dim_node))
        return mapped.view(*orig_shape[:-1], -1)
