# atlas/cpu_block.py
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AtlasLayer:
    layer: int

    # tree view (can be left as None initially and assigned later when needed)
    dst_nodes_tree: Optional[Tensor] = None
    dst_ts_tree:    Optional[Tensor] = None
    dst_feat_tree:  Optional[Tensor] = None

    src_nodes_tree: Optional[Tensor] = None
    src_ts_tree:    Optional[Tensor] = None
    src_feat_tree:  Optional[Tensor] = None
    edge_ids_tree:  Optional[Tensor] = None
    edge_feat_tree: Optional[Tensor] = None

    # flat view
    dst_nodes: Optional[Tensor] = None  # [N_l]
    dst_ts:    Optional[Tensor] = None  # [N_l]

    # dst_index: Optional[Tensor] = None  # [N_l]
    dst_feat:  Optional[Tensor] = None  # [N_l, Dn]


    src_nodes: Optional[Tensor] = None  # [N_l, K]
    src_ts:    Optional[Tensor] = None  # [N_l, K]
    src_feat:  Optional[Tensor] = None  # [N_l, K, Dn]
    # src_index:  Optional[Tensor] = None  # [N_l, K]

    edge_ids:  Optional[Tensor] = None  # [N_l, K]
    edge_feat: Optional[Tensor] = None  # [N_l, K, De]
    # edge_index: Optional[Tensor] = None  # [N_l, K]

@dataclass
class AtlasCPUBlock:
    batch_size:    int
    num_layers:    int
    num_neighbors: int

    root_src: Tensor
    root_dst: Tensor
    root_neg: Tensor
    root_ts:  Tensor

    root_src_feat: Tensor
    root_dst_feat: Tensor
    root_neg_feat: Tensor
    root_edge_feat: Tensor

    atlas_layers: List[AtlasLayer]


    unique_nodes: Optional[Tensor] = None
    unique_edges: Optional[Tensor] = None
    nodes_feat_pool: Optional[Tensor] = None
    edges_feat_pool: Optional[Tensor] = None


def _pin(x: Tensor) -> Tensor:
    if x is None or not isinstance(x, Tensor):
        return x
    if x.is_cuda:
        return x
    if not torch.cuda.is_available():
        return x
    return x.pin_memory()
