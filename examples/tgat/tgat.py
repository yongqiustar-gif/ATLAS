# tgat_atlas.py
from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import List, Optional

from atlas import AtlasGPUBlock, AtlasGPULayer
from atlas.nn_layers import AtlasTemporalAttnLayer
import os
import sys

# Add the examples directory (where support.py is located) to sys.path
THIS_DIR = os.path.dirname(__file__)              # /home/atlas/examples/tgat
EXAMPLES_DIR = os.path.dirname(THIS_DIR)          # /home/atlas/examples
if EXAMPLES_DIR not in sys.path:
    sys.path.append(EXAMPLES_DIR)

import support


class AtlasTGAT(nn.Module):
    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_time: int,
        dim_embed: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            AtlasTemporalAttnLayer(
                dim_node=dim_node,
                dim_edge=dim_edge,
                dim_time=dim_time,
                dim_out=dim_embed,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.embed_to_node = nn.Linear(dim_embed, dim_node)
        self.edge_predictor = support.EdgePredictor(dim=dim_embed)

    def forward(self, block: AtlasGPUBlock) -> tuple[Tensor, Tensor]:
        """
        :param block: AtlasGPUBlock
        :return: (pos_scores, neg_scores) [B], [B]
        """
        B = block.batch_size
        L = block.num_layers

        next_src_embed: Optional[Tensor] = None
        aggregated_root: Optional[Tensor] = None
        root_layers = 0
        group = 3 * B

        for l in reversed(range(L)):
            glayer: AtlasGPULayer = block.layers[l]

            if next_src_embed is not None:
                glayer.src_feat = next_src_embed

            h_l = self.layers[l](glayer)        # [N_l, dim_embed]

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

        pos_scores = self.edge_predictor(src_h, dst_h)   # [B]
        neg_scores = self.edge_predictor(src_h, neg_h)   # [B]
        return pos_scores, neg_scores
