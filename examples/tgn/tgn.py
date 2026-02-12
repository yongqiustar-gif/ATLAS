# tgat_atlas.py
from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import List, Optional, Tuple

import atlas as at
from atlas import AtlasGPUBlock, AtlasGPULayer
from atlas.nn_layers import AtlasTemporalMemoryAttnLayer
import os
import sys

# Add the examples directory (where support.py is located) to sys.path
THIS_DIR = os.path.dirname(__file__)              # /home/atlas/examples/tgat
EXAMPLES_DIR = os.path.dirname(THIS_DIR)          # /home/atlas/examples
if EXAMPLES_DIR not in sys.path:
    sys.path.append(EXAMPLES_DIR)

import support


class AtlasTGN(nn.Module):
    def __init__(
        self, 
        g: at.TGraph,
        dim_node: int,
        dim_edge: int,
        dim_time: int,
        dim_embed: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.g = g
        self.dim_edge = dim_edge
        self.dim_node = dim_node
        self.attn_dim = dim_embed if dim_node != dim_embed else dim_node
        self.nfeat_map = None if dim_node == self.attn_dim else nn.Linear(dim_node, self.attn_dim)

        mem_init = torch.zeros(g.num_nodes(), dim_embed, dtype=torch.float32, device=device)
        ts_init = torch.zeros(g.num_nodes(), dtype=torch.float32, device=device)
        self.register_buffer("memory", mem_init)
        self.register_buffer("memory_ts", ts_init)
        self.num_layers = num_layers
        self.mem_cell = nn.GRUCell(2 * dim_embed + dim_edge + dim_time, dim_embed)
        self.mem_time_encode = at.nn_layers.TimeEncode(dim_time)
        self.attn = nn.ModuleList([
            AtlasTemporalMemoryAttnLayer(
                dim_node=self.attn_dim,
                dim_edge=dim_edge,
                dim_time=dim_time,
                dim_embed=dim_embed,
                dim_out=dim_embed,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.embed_to_node = nn.Linear(dim_embed, self.attn_dim)

        self.edge_predictor = support.EdgePredictor(dim=dim_embed)
        

    def forward(self, block: AtlasGPUBlock) -> tuple[Tensor, Tensor]:
        B = block.batch_size
        L = block.num_layers

        next_src_embed: Optional[Tensor] = None
        aggregated_root: Optional[Tensor] = None
        root_layers = 0
        group = 3 * B
        for l in reversed(range(L)):
            glayer: AtlasGPULayer = block.layers[l]

            if next_src_embed is not None:
                src_backup = glayer.src_feat
                glayer.src_feat = next_src_embed
                map_src = False
            else:
                src_backup = None
                map_src = True

            dst_backup, src_feat_backup = self._map_layer_feats(glayer, map_src=map_src)

            h_l = self.attn[l](glayer, self.memory)        # [N_l, dim_embed]

            self._restore_layer_feats(glayer, dst_backup, src_feat_backup)
            if src_backup is not None:
                glayer.src_feat = src_backup

            N_l = h_l.shape[0]
            fanout_hint = glayer.src_feat.shape[1] if glayer.src_feat.dim() >= 2 else None
            if fanout_hint is not None and fanout_hint > 0 and N_l == group * fanout_hint:
                fanout = fanout_hint
            else:
                assert group > 0 and N_l % group == 0, (
                    f"invalid layer shape: N_l={N_l}, group={group}, fanout_hint={fanout_hint}"
                )
                fanout = max(1, N_l // group)

            # reshape: [3B, fanout, dim_embed], reduce along fanout to get root representation
            h_root = h_l.reshape(group, fanout, -1).mean(dim=1)
            if aggregated_root is None:
                aggregated_root = h_root.clone()
            else:
                aggregated_root.add_(h_root)
            root_layers += 1

            if l > 0:
                parent = block.layers[l - 1]
                projected = self.embed_to_node(h_l)
                next_src_embed = projected.reshape(
                    parent.src_feat.shape[0], parent.src_feat.shape[1], -1
                )
            else:
                next_src_embed = None


        if aggregated_root is None:
            raise RuntimeError("AtlasTGAT forward received empty block layers")
        h_total = aggregated_root if root_layers == 1 else aggregated_root / root_layers

        src_h = h_total[0:B]
        dst_h = h_total[B:2 * B]
        neg_h = h_total[2 * B:3 * B]

        update_nodes = block.layers[0].dst_nodes[0:2*B].clone()
        update_msgs = h_total[0: 2*B]  # [2B, dim_embed]
        mail_ts = block.root_ts.repeat(2).to(update_msgs.device)
        prev_ts = self.memory_ts[update_nodes].detach()
        delta_t = torch.clamp(mail_ts - prev_ts, min=0.0)
        time_feat = self.mem_time_encode(delta_t)
        if block.root_edge_feat is not None:
            edge_feat = block.root_edge_feat.repeat(2, 1)
        else:
            edge_feat = torch.zeros(2 * B, self.dim_edge, device=update_msgs.device)

        prev_mem = self.memory[update_nodes].detach()
        new_mem = self.mem_cell(
            torch.cat([
                update_msgs,
                prev_mem,
                time_feat,
                edge_feat
            ], dim=1),
            prev_mem
        )
        self.memory[update_nodes] = new_mem.detach()
        self.memory_ts[update_nodes] = mail_ts.detach()

        pos_scores = self.edge_predictor(src_h, dst_h)   # [B]
        neg_scores = self.edge_predictor(src_h, neg_h)   # [B]
        return pos_scores, neg_scores

    def _map_layer_feats(
        self,
        layer: AtlasGPULayer,
        *,
        map_src: bool,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if self.nfeat_map is None:
            return None, None

        dst_backup: Optional[Tensor] = None
        src_backup: Optional[Tensor] = None

        if layer.dst_feat is not None and layer.dst_feat.shape[-1] == self.dim_node:
            dst_backup = layer.dst_feat
            layer.dst_feat = self._map_feat(layer.dst_feat)

        if map_src and layer.src_feat is not None and layer.src_feat.shape[-1] == self.dim_node:
            src_backup = layer.src_feat
            layer.src_feat = self._map_feat(layer.src_feat)

        return dst_backup, src_backup

    def _restore_layer_feats(
        self,
        layer: AtlasGPULayer,
        dst_backup: Optional[Tensor],
        src_backup: Optional[Tensor],
    ) -> None:
        if dst_backup is not None:
            layer.dst_feat = dst_backup
        if src_backup is not None:
            layer.src_feat = src_backup

    def _map_feat(self, feat: Tensor) -> Tensor:
        assert self.nfeat_map is not None
        orig_shape = feat.shape
        mapped = self.nfeat_map(feat.reshape(-1, self.dim_node))
        return mapped.view(*orig_shape[:-1], -1)
