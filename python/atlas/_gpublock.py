# atlas/gpu_block.py
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Optional

from ._cpublock import AtlasCPUBlock
from . import _c

from torch.cuda import nvtx

@dataclass
class AtlasGPULayer:

    layer: int

    # [N_l]
    dst_nodes: Tensor      # int64
    dst_ts:    Tensor      # float32
    dst_feat:  Tensor      # float32 [N_l, Dn]

    # [N_l, K]
    src_nodes: Tensor      # int64
    src_ts:    Tensor      # float32
    src_feat:  Tensor      # float32 [N_l, K, Dn]
    edge_ids:  Tensor      # int64
    edge_feat: Tensor      # float32 [N_l, K, De]

    # dst_index: Optional[Tensor] = None  # [N_l]
    # src_index: Optional[Tensor] = None  # [N_l, K]
    # edge_index: Optional[Tensor] = None  # [N_l, K]


@dataclass
class AtlasGPUBlock:
    """
    Multi-layer view of one batch on GPU.
    Used by model.forward(atlas_gpu_block).
    """
    batch_size:    int
    num_layers:    int
    num_neighbors: int

    # root triples (kept on GPU for easier loss computation)
    root_src: Tensor        # [B], int64
    root_dst: Tensor        # [B]
    root_neg: Tensor        # [B]
    root_ts:  Tensor        # [B], float32

    root_src_feat: Tensor   # [B, Dn]
    root_dst_feat: Tensor   # [B, Dn]
    root_neg_feat: Tensor   # [B, Dn]
    root_edge_feat: Tensor  # [B, De]

    # per-layer data
    layers: List[AtlasGPULayer]

    unique_nodes: Optional[Tensor] = None
    unique_edges: Optional[Tensor] = None
    nodes_feat_pool: Optional[Tensor] = None
    edges_feat_pool: Optional[Tensor] = None


def cpu_to_gpu_block(cpu_block: "AtlasCPUBlock", device: torch.device) -> AtlasGPUBlock:
    """
    Copy AtlasCPUBlock to GPU and build AtlasGPUBlock.
    (Assumes tensors in cpu_block are on CPU, ideally pinned.)
    """
    def to_dev(x: Tensor) -> Tensor:
        return x.to(device, non_blocking=True)

    gpu_layers: list[AtlasGPULayer] = []
    for layer in cpu_block.atlas_layers:
        if layer.dst_feat == None:
                        gpu_layers.append(
            AtlasGPULayer(
                layer=layer.layer,
                dst_nodes=to_dev(layer.dst_nodes),
                dst_ts=to_dev(layer.dst_ts),
                dst_feat=None,
                src_nodes=to_dev(layer.src_nodes),
                src_ts=to_dev(layer.src_ts),
                src_feat=None,
                edge_ids=to_dev(layer.edge_ids),
                edge_feat=None,
            )
            )
        else:
            gpu_layers.append(
            AtlasGPULayer(
                layer=layer.layer,
                dst_nodes=to_dev(layer.dst_nodes),
                dst_ts=to_dev(layer.dst_ts),
                dst_feat=to_dev(layer.dst_feat),
                src_nodes=to_dev(layer.src_nodes),
                src_ts=to_dev(layer.src_ts),
                src_feat=to_dev(layer.src_feat),
                edge_ids=to_dev(layer.edge_ids),
                edge_feat=to_dev(layer.edge_feat),
            )
            )

    return AtlasGPUBlock(
        batch_size=cpu_block.batch_size,
        num_layers=cpu_block.num_layers,
        num_neighbors=cpu_block.num_neighbors,
        root_src=to_dev(cpu_block.root_src),
        root_dst=to_dev(cpu_block.root_dst),
        root_neg=to_dev(cpu_block.root_neg),
        root_ts=to_dev(cpu_block.root_ts),
        root_src_feat=to_dev(cpu_block.root_src_feat),
        root_dst_feat=to_dev(cpu_block.root_dst_feat),
        root_neg_feat=to_dev(cpu_block.root_neg_feat),
        root_edge_feat=to_dev(cpu_block.root_edge_feat),
        layers=gpu_layers,
        unique_nodes=to_dev(cpu_block.unique_nodes) if cpu_block.unique_nodes is not None else None,
        unique_edges=to_dev(cpu_block.unique_edges) if cpu_block.unique_edges is not None else None,
        nodes_feat_pool=to_dev(cpu_block.nodes_feat_pool) if cpu_block.nodes_feat_pool is not None else None,
        edges_feat_pool=to_dev(cpu_block.edges_feat_pool) if cpu_block.edges_feat_pool is not None else None,
    )



def cpu_to_gpu_block_index(cpu_block: "AtlasCPUBlock", device: torch.device) -> AtlasGPUBlock:
    """
    Copy index tensors in AtlasCPUBlock to GPU and build AtlasGPUBlock.
    (Assumes tensors in cpu_block are on CPU, ideally pinned.)
    """
    def to_dev(x: Tensor) -> Tensor:
        return x.to(device, non_blocking=True)

    gpu_layers: list[AtlasGPULayer] = []
    for layer in cpu_block.atlas_layers:
        gpu_layers.append(
            AtlasGPULayer(
                layer=layer.layer,
                dst_nodes=to_dev(layer.dst_nodes),
                dst_ts=to_dev(layer.dst_ts),
                dst_feat=None,
                src_nodes=to_dev(layer.src_nodes),
                src_ts=to_dev(layer.src_ts),
                src_feat=None,
                edge_ids=to_dev(layer.edge_ids),
                edge_feat=None,
            )
        )

    return AtlasGPUBlock(
        batch_size=cpu_block.batch_size,
        num_layers=cpu_block.num_layers,
        num_neighbors=cpu_block.num_neighbors,
        root_src=to_dev(cpu_block.root_src),
        root_dst=to_dev(cpu_block.root_dst),
        root_neg=to_dev(cpu_block.root_neg),
        root_ts=to_dev(cpu_block.root_ts),
        root_src_feat=to_dev(cpu_block.root_src_feat) if cpu_block.root_src_feat is not None else None,
        root_dst_feat=to_dev(cpu_block.root_dst_feat) if cpu_block.root_dst_feat is not None else None,
        root_neg_feat=to_dev(cpu_block.root_neg_feat) if cpu_block.root_neg_feat is not None else None,
        root_edge_feat=to_dev(cpu_block.root_edge_feat) if cpu_block.root_edge_feat is not None else None,
        layers=gpu_layers,
        unique_nodes=to_dev(cpu_block.unique_nodes) if cpu_block.unique_nodes is not None else None,
        unique_edges=to_dev(cpu_block.unique_edges) if cpu_block.unique_edges is not None else None,
        nodes_feat_pool=to_dev(cpu_block.nodes_feat_pool) if cpu_block.nodes_feat_pool is not None else None,
        edges_feat_pool=to_dev(cpu_block.edges_feat_pool) if cpu_block.edges_feat_pool is not None else None,
        
    )


def recover_block(block: AtlasGPUBlock) -> AtlasGPUBlock:
    """
    Replace indices in block with features.
    Requires nodes_feat_pool and edges_feat_pool to be present in block beforehand.
    """
    assert block.nodes_feat_pool is not None
    assert block.edges_feat_pool is not None
    nvtx.range_push("recover_block")
    for layer in block.layers:
        nvtx.range_push(f"layer_{layer.layer}")
        nvtx.range_push("index2feat_dst")
        layer.dst_feat = index2feat(layer.dst_nodes, block.nodes_feat_pool, block.unique_nodes)
        nvtx.range_pop()
        nvtx.range_push("index2feat_src")
        layer.src_feat = index2feat(layer.src_nodes, block.nodes_feat_pool, block.unique_nodes)
        nvtx.range_pop()
        nvtx.range_push("index2feat_edge")
        layer.edge_feat = index2feat(layer.edge_ids, block.edges_feat_pool, block.unique_edges)
        nvtx.range_pop()
        nvtx.range_pop()
    nvtx.range_pop()
    return block

def index2feat(index: Tensor, feat_pool: Tensor, unique_nodes: Tensor) -> Tensor:
    """
    Gather features from feat_pool using index.
    index: [N] or [N, K], int64
    feat_pool: [V, D]
    unique_nodes: [V]
    returns: [N, D] or [N, K, D]
    """
    feat_dim = feat_pool.shape[-1]
    out_shape = (*index.shape, feat_dim)

    if unique_nodes is None or unique_nodes.numel() == 0 or index.numel() == 0:
        return torch.zeros(out_shape, device=feat_pool.device, dtype=feat_pool.dtype)

    if index.is_cuda:
        return _c.recover_feat_from_pool(index, unique_nodes, feat_pool)

    out = torch.zeros(out_shape, device=feat_pool.device, dtype=feat_pool.dtype)
    flat_index = index.reshape(-1)
    valid_mask = flat_index >= 0
    if not valid_mask.any():
        return out

    flat_valid = flat_index[valid_mask]
    pos = torch.searchsorted(unique_nodes, flat_valid)
    in_range = pos < unique_nodes.shape[0]
    if not in_range.any():
        return out

    pos = pos[in_range]
    flat_valid = flat_valid[in_range]
    match = unique_nodes[pos] == flat_valid
    if not match.any():
        return out

    pos = pos[match]
    assign_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)[in_range][match]
    gathered = feat_pool[pos]

    out_flat = out.view(-1, feat_dim)
    out_flat[assign_idx] = gathered
    return out_flat.view_as(out)