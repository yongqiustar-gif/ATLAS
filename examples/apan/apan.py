"""Atlas implementation of APAN aligned with the original TGLite logic."""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

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


class AttnMemoryUpdater(nn.Module):
    def __init__(
        self,
        dim_mem: int,
        dim_msg: int,
        dim_time: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert dim_mem % num_heads == 0, "dim_mem must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim_mem // num_heads
        self.w_q = nn.Linear(dim_mem, dim_mem)
        self.w_k = nn.Linear(dim_msg + dim_time, dim_mem)
        self.w_v = nn.Linear(dim_msg + dim_time, dim_mem)
        self.mlp = nn.Linear(dim_mem, dim_mem)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_mem)
        self.time_encode = at.nn_layers.TimeEncode(dim_time)

    def forward(
        self,
        prev_mem: Tensor,
        mail: Tensor,
        mail_ts: Tensor,
        curr_ts: Tensor,
        mask: Tensor,
    ) -> Tensor:
        # mail: [N, S, dim_msg], mail_ts: [N, S], mask: [N, S]
        N, slots, _ = mail.shape
        device = prev_mem.device
        if slots == 0:
            return prev_mem

        dt = curr_ts.unsqueeze(1) - mail_ts
        time_feat = self.time_encode(dt.reshape(-1)).reshape(N, slots, -1)
        msg = torch.cat([mail, time_feat], dim=-1)

        Q = self.w_q(prev_mem).reshape(N, self.num_heads, self.head_dim)
        K = self.w_k(msg).reshape(N, slots, self.num_heads, self.head_dim)
        V = self.w_v(msg).reshape(N, slots, self.num_heads, self.head_dim)

        attn = (Q.unsqueeze(1) * K).sum(-1) / math.sqrt(self.head_dim)
        fallback_mask = mask.clone()
        empty_rows = ~mask.any(dim=1)
        fallback_mask[:, 0] = torch.where(
            empty_rows,
            torch.ones_like(fallback_mask[:, 0], dtype=fallback_mask.dtype),
            fallback_mask[:, 0],
        )
        attn = attn.masked_fill(~fallback_mask.unsqueeze(-1), float("-inf"))
        attn = torch.softmax(attn, dim=1)
        attn = torch.where(mask.unsqueeze(-1), attn, torch.zeros_like(attn))

        attn_sum = attn.sum(dim=1, keepdim=True)
        attn = attn / attn_sum.clamp_min(1e-6)
        attn = self.dropout(attn)

        out = (attn.unsqueeze(-1) * V).sum(dim=1)
        out = out.reshape(N, -1)
        out = self.layer_norm(out + prev_mem)
        out = self.mlp(out)
        out = self.dropout(torch.relu(out))
        return out


class AtlasApan(nn.Module):
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
        mail_slots: int = 10,
    ) -> None:
        super().__init__()
        self.g = g
        self.dim_embed = dim_embed
        self.dim_edge = dim_edge
        self.dim_msg = 2 * dim_embed + dim_edge
        self.num_layers = num_layers
        self.mail_slots_cfg = max(1, mail_slots)

        mem_init = torch.zeros(g.num_nodes(), dim_embed, dtype=torch.float32, device=device)
        ts_init = torch.zeros(g.num_nodes(), dtype=torch.float32, device=device)
        self.register_buffer("memory", mem_init)
        self.register_buffer("memory_ts", ts_init)

        self.mail: Optional[Tensor] = None
        self.mail_ts: Optional[Tensor] = None
        self.mail_ptr: Optional[Tensor] = None
        self.mail_count: Optional[Tensor] = None
        self.mail_slots: Optional[int] = None

        self.mem_updater = AttnMemoryUpdater(
            dim_mem=dim_embed,
            dim_msg=self.dim_msg,
            dim_time=dim_time,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.edge_predictor = support.EdgePredictor(dim_embed)

    def forward(self, block: AtlasGPUBlock) -> tuple[Tensor, Tensor]:
        B = block.batch_size
        if B == 0:
            raise RuntimeError("AtlasApan received empty block")

        layer0 = block.layers[0]
        nodes = layer0.dst_nodes.clone()
        device = nodes.device
        self._ensure_mailbox(layer0, device)

        prev_mem = self.memory[nodes].detach()
        curr_ts = block.root_ts.repeat(3).to(device)
        mail, mail_ts, mask = self._gather_mail(nodes)
        updated = self.mem_updater(prev_mem, mail, mail_ts, curr_ts, mask)

        pos_nodes = nodes[: 2 * B]
        pos_ts = block.root_ts.repeat(2).to(device)
        self.memory[pos_nodes] = updated[: 2 * B].detach()
        self.memory_ts[pos_nodes] = pos_ts.detach()

        root_mail, root_mail_ts = self._build_root_mail(updated, block)
        self._propagate_mail(block, root_mail, root_mail_ts)

        src_h = updated[0:B]
        dst_h = updated[B : 2 * B]
        neg_h = updated[2 * B : 3 * B]

        pos_scores = self.edge_predictor(src_h, dst_h)
        neg_scores = self.edge_predictor(src_h, neg_h)
        return pos_scores, neg_scores

    def _ensure_mailbox(self, layer0: AtlasGPULayer, device: torch.device) -> None:
        del layer0  # fanout no longer dictates mailbox depth
        slots = self.mail_slots_cfg
        if self.mail is not None and self.mail_slots == slots:
            return
        num_nodes = self.g.num_nodes()
        self.mail_slots = slots
        self.mail = torch.zeros(num_nodes, slots, self.dim_msg, device=device)
        self.mail_ts = torch.zeros(num_nodes, slots, device=device)
        self.mail_ptr = torch.zeros(num_nodes, dtype=torch.long, device=device)
        self.mail_count = torch.zeros(num_nodes, dtype=torch.long, device=device)

    def _gather_mail(self, nodes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        assert self.mail is not None and self.mail_ts is not None
        assert self.mail_ptr is not None and self.mail_count is not None
        slots = self.mail.shape[1]
        mail = self.mail[nodes]
        mail_ts = self.mail_ts[nodes]
        ptr = self.mail_ptr[nodes]
        counts = self.mail_count[nodes]

        order = (ptr.unsqueeze(1) - torch.arange(slots, device=nodes.device) - 1) % slots
        order_exp = order.unsqueeze(-1).expand(-1, -1, self.dim_msg)
        mail = torch.gather(mail, 1, order_exp)
        mail_ts = torch.gather(mail_ts, 1, order)
        mask = torch.arange(slots, device=nodes.device).unsqueeze(0) < counts.unsqueeze(1)
        return mail, mail_ts, mask

    def _build_root_mail(self, updated: Tensor, block: AtlasGPUBlock) -> tuple[Tensor, Tensor]:
        B = block.batch_size
        device = updated.device
        mem_src = updated[0:B]
        mem_dst = updated[B : 2 * B]
        if self.dim_edge > 0 and block.root_edge_feat is not None:
            edge_feat = block.root_edge_feat.to(device)
        else:
            edge_feat = torch.zeros(B, self.dim_edge, device=device)

        if self.dim_edge > 0:
            src_mail = torch.cat([mem_src, mem_dst, edge_feat], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src, edge_feat], dim=1)
        else:
            src_mail = torch.cat([mem_src, mem_dst], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src], dim=1)

        mails = torch.cat([src_mail, dst_mail], dim=0)
        ts = block.root_ts.repeat(2).to(device)
        return mails.detach(), ts.detach()

    def _propagate_mail(self, block: AtlasGPUBlock, root_mail: Tensor, root_ts: Tensor) -> None:
        if block.num_layers == 0 or root_mail.numel() == 0:
            return
        device = root_mail.device
        msg_dim = self.dim_msg

        layer_mail = torch.zeros(block.layers[0].dst_nodes.shape[0], msg_dim, device=device)
        layer_ts = torch.zeros(block.layers[0].dst_nodes.shape[0], device=device)
        span = min(layer_mail.shape[0], root_mail.shape[0])
        layer_mail[:span] = root_mail[:span]
        layer_ts[:span] = root_ts[:span]

        num_layers = int(block.num_layers)
        for layer_idx in range(num_layers):
            layer = block.layers[layer_idx]
            if layer.src_nodes is None or layer.src_nodes.numel() == 0:
                continue
            fanout = layer.src_nodes.shape[1] if layer.src_nodes.dim() >= 2 else 1
            mail_flat = layer_mail.unsqueeze(1).expand(-1, fanout, -1).reshape(-1, msg_dim)
            ts_flat = layer_ts.unsqueeze(1).expand(-1, fanout).reshape(-1)
            src_flat = layer.src_nodes.reshape(-1)
            valid = src_flat >= 0
            agg_mail, agg_ts, counts = self._aggregate_messages(
                src_flat, mail_flat, ts_flat, valid
            )
            self._store_mail_entries(agg_mail, agg_ts, counts)

            mail_next = mail_flat.clone()
            ts_next = ts_flat.clone()
            mail_next[~valid] = 0.0
            ts_next[~valid] = 0.0

            if layer_idx + 1 < num_layers:
                next_layer = block.layers[layer_idx + 1]
                expected = next_layer.dst_nodes.shape[0]
                if mail_next.shape[0] < expected:
                    pad = expected - mail_next.shape[0]
                    layer_mail = torch.cat(
                        [mail_next, torch.zeros(pad, msg_dim, device=device)],
                        dim=0,
                    )
                    layer_ts = torch.cat(
                        [ts_next, torch.zeros(pad, device=device)],
                        dim=0,
                    )
                else:
                    layer_mail = mail_next[:expected]
                    layer_ts = ts_next[:expected]

    def _aggregate_messages(
        self,
        nodes: Tensor,
        mail: Tensor,
        ts: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        num_nodes = self.g.num_nodes()
        device = mail.device
        if nodes.numel() == 0:
            return (
                torch.zeros(num_nodes, self.dim_msg, device=device),
                torch.zeros(num_nodes, device=device),
                torch.zeros(num_nodes, device=device),
            )

        nodes_safe = torch.where(valid, nodes, torch.zeros_like(nodes))
        weights = valid.to(mail.dtype)
        mail_weighted = mail * weights.unsqueeze(-1)
        ts_weighted = ts * weights

        agg_mail = torch.zeros(num_nodes, self.dim_msg, device=device)
        agg_ts = torch.zeros(num_nodes, device=device)
        counts = torch.zeros(num_nodes, device=device)
        agg_mail.index_add_(0, nodes_safe.long(), mail_weighted)
        agg_ts.index_add_(0, nodes_safe.long(), ts_weighted)
        counts.index_add_(0, nodes_safe.long(), weights)

        denom = counts.clamp_min(1.0)
        agg_mail = agg_mail / denom.unsqueeze(-1)
        agg_ts = agg_ts / denom
        return agg_mail, agg_ts, counts

    def _store_mail_entries(
        self,
        agg_mail: Tensor,
        agg_ts: Tensor,
        counts: Tensor,
    ) -> None:
        assert self.mail is not None and self.mail_ts is not None
        assert self.mail_ptr is not None and self.mail_count is not None
        slots = self.mail.shape[1]
        active = counts > 0
        node_idx = torch.arange(counts.shape[0], device=counts.device)
        ptr = self.mail_ptr

        curr_mail = self.mail[node_idx, ptr]
        curr_ts = self.mail_ts[node_idx, ptr]
        agg_mail = agg_mail.detach()
        agg_ts = agg_ts.detach()

        mail_update = torch.where(active.unsqueeze(-1), agg_mail, curr_mail)
        ts_update = torch.where(active, agg_ts, curr_ts)
        self.mail[node_idx, ptr] = mail_update
        self.mail_ts[node_idx, ptr] = ts_update

        inc = active.long()
        self.mail_ptr[:] = (ptr + inc) % slots
        self.mail_count[:] = torch.clamp(self.mail_count + inc, max=slots)
