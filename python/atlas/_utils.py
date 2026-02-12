import os
from typing import Callable, List

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._block import TBlock
    from ._graph import TGraph

import numpy as np
import torch

from . import _c
from ._core import TError

from torch.cuda import nvtx

def get_num_cpus(default=16) -> int:
    cpus = os.cpu_count()
    return default if cpus is None else cpus


def check_edges_times(edges: np.ndarray, times: np.ndarray):
    if edges.shape[0] != times.shape[0]:
        raise TError("edge list and timestamps must have same leading dimension")
    if edges.shape[1] != 2:
        raise TError("edge list must have only 2 columns")
    if edges.dtype != np.int32:
        raise TError("currently only supports int32 node/edge ids")
    if times.dtype != np.float32:
        raise TError("currently only supports float32 timestamps")


def check_num_nodes(edges: np.ndarray, num_nodes: int = None) -> int:
    """Returns the number of nodes in the graph represented by the given edges
    
    :raises TErrror: if the specified number of nodes is less than or equal to the number of distinct nodes present in the edges
    """
    max_nid = int(edges.max())
    num_nodes = max_nid + 1 \
        if num_nodes is None else num_nodes
    if num_nodes <= max_nid:
        raise TError("number of nodes must be greater than max node id")
    return num_nodes


def create_tcsr(edges: np.ndarray, times: np.ndarray, num_nodes: int = None):
    check_edges_times(edges, times)
    num_nodes = check_num_nodes(edges, num_nodes)
    return _c.create_tcsr(edges, times, num_nodes)


from ._cpublock import AtlasCPUBlock, AtlasLayer, _pin
from ._sampler import FixedTSampler

def pinned_gather(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Sample from feat on CPU by idx, and return a pinned-memory tensor

    :param feat: [N, D]
    :param idx:  [M]
    :return:     [M, D] pin memory
    """
    if torch.cuda.is_available():
        return _c.pinned_gather(feat, idx)
    return feat[idx]




def _fixed_sample_one_layer(
    g,
    sampler: FixedTSampler,
    nodes: torch.Tensor,         # [N_l]
    times: torch.Tensor,         # [N_l]
    nfeat: torch.Tensor,         # [N, Dn] cpu
    efeat: torch.Tensor,         # [E, De] cpu
    layer_idx: int,
    num_neighbors: int,
) -> AtlasLayer:
    """
    One-layer sampling -> AtlasLayer (flat view)
    """
    device = torch.device("cpu")
    nvtx.range_push("fixed sample one layer")

    nodes = nodes.to(device, dtype=torch.long)
    times = times.to(device, dtype=torch.float32)

    N_l = nodes.shape[0]
    K = num_neighbors

    # --------- call C++ sampler ----------
    nodes_np = nodes.to(dtype=torch.int32).cpu().numpy()
    times_np = times.cpu().numpy()  # already float32

    pad_mask = (nodes_np < 0)        # padding nodes do not participate in real sampling
    nodes_q = nodes_np.copy()
    times_q = times_np.copy()
    nodes_q[pad_mask] = 0
    times_q[pad_mask] = 0.0

    block = sampler.sample_block(g, nodes_q, times_q)

    dstindex_np = block.copy_dstindex()   # [N_l * K]
    srcnodes_np = block.copy_srcnodes()   # [N_l * K]
    eids_np     = block.copy_eid()
    ets_np      = block.copy_ets()

    nvtx.range_pop()
    assert dstindex_np.shape[0] == N_l * K, "fixed sampler should output N_l*K edges"

    nvtx.range_push("padding sampled data")
    # --------- fill padding positions with -1 ----------
    if pad_mask.any():
        idx = np.where(pad_mask)[0]
        src_pad = srcnodes_np.reshape(N_l, K)
        eids_pad = eids_np.reshape(N_l, K)
        ets_pad  = ets_np.reshape(N_l, K)
        src_pad[idx, :] = -1
        eids_pad[idx, :] = -1
        ets_pad[idx, :] = 0.0
        srcnodes_np = src_pad.reshape(-1)
        eids_np     = eids_pad.reshape(-1)
        ets_np      = ets_pad.reshape(-1)

    nvtx.range_pop()
    nvtx.range_push("convert to torch & reshape")
    nvtx.range_push("pin dst nodes & times")
    dst_nodes = _pin(nodes)        # [N_l]
    dst_ts    = _pin(times)        # [N_l]
    nvtx.range_push("pin dst feat")
    # dst_feat  = _pin(nfeat[dst_nodes.clamp(min=0)])   # treat padding (-1) as node 0 for safe gather
    dst_feat = pinned_gather(nfeat, dst_nodes.clamp(min=0)).view(N_l, -1)
    nvtx.range_pop()
    nvtx.range_pop()
    nvtx.range_push("pin src nodes, times & edge ids")
    src_nodes = _pin(torch.from_numpy(srcnodes_np).to(device=device, dtype=torch.long).view(N_l, K))
    src_ts    = _pin(torch.from_numpy(ets_np).to(device=device, dtype=torch.float32).view(N_l, K))
    edge_ids  = _pin(torch.from_numpy(eids_np).to(device=device, dtype=torch.long).view(N_l, K))

    src_nodes_clamp = src_nodes
    src_nodes_clamp[src_nodes_clamp < 0] = 0
    nvtx.range_push("pin src feat")
    # src_feat = _pin(nfeat[src_nodes_clamp])       # [N_l, K, Dn]
    src_feat = pinned_gather(nfeat, src_nodes_clamp.reshape(-1)).view(N_l, K, -1)
    nvtx.range_pop()
    nvtx.range_push("clamp edge ids & pin edge feat")
    edge_ids_clamp = edge_ids
    edge_ids_clamp[edge_ids_clamp < 0] = 0
    nvtx.range_pop()
    nvtx.range_push("pin edge feat")
    # edge_feat = _pin(efeat[edge_ids_clamp])       # [N_l, K, De]
    edge_feat = pinned_gather(efeat, edge_ids_clamp.reshape(-1)).view(N_l, K, -1)
    nvtx.range_pop()
    nvtx.range_pop()

    nvtx.range_pop()
    return AtlasLayer(
        layer=layer_idx,
        dst_nodes=dst_nodes,
        dst_ts=dst_ts,
        dst_feat=dst_feat,
        src_nodes=src_nodes,
        src_ts=src_ts,
        src_feat=src_feat,
        edge_ids=edge_ids,
        edge_feat=edge_feat,
    )


def make_batch(
    g: "TGraph",
    start: int,
    end: int,
    neg_sampler: Callable[[int], np.ndarray],
    sampler: FixedTSampler,
    num_layers: int,
    num_neighbors: int,
) -> AtlasCPUBlock:
    """
    Build AtlasCPUBlock from edges in [start, end) with sampler
    """
    device = torch.device("cpu")

    nvtx.range_push("make edges")
    # ---- fetch current batch edges ----
    edges = g._edges[start:end]     # [B, 2]
    ts    = g._times[start:end]        # [B]

    if isinstance(edges, np.ndarray):
        edges = torch.from_numpy(edges)
    if isinstance(ts, np.ndarray):
        ts = torch.from_numpy(ts)

    B = edges.shape[0]
    src = edges[:, 0].to(device, dtype=torch.long)
    dst = edges[:, 1].to(device, dtype=torch.long)
    ts  = ts.to(device, dtype=torch.float32)

    # ---- root triples ----
    root_src = src
    root_dst = dst
    root_ts  = ts

    neg_np   = neg_sampler(B).astype(np.int64)
    root_neg = torch.from_numpy(neg_np).to(device=device, dtype=torch.long)
    nvtx.range_pop()  # make edges

    nvtx.range_push("make features")
    # ---- feature tables ----
    nfeat = g.nfeat
    efeat = g.efeat
    if isinstance(nfeat, np.ndarray):
        nfeat = torch.from_numpy(nfeat)
    if isinstance(efeat, np.ndarray):
        efeat = torch.from_numpy(efeat)
    nfeat = nfeat.to(device)
    efeat = efeat.to(device)

    Dn = nfeat.shape[1]
    De = efeat.shape[1]

    eid_pos = torch.arange(start, end, dtype=torch.long, device=device)

    root_src_feat = nfeat[root_src]
    root_dst_feat = nfeat[root_dst]
    root_neg_feat = nfeat[root_neg]
    root_edge_feat = efeat[eid_pos]
    nvtx.range_pop()  # make features

    nvtx.range_push("multi-layer sampling")
    # ---- multi-layer sampling ----
    atlas_layers: List[AtlasLayer] = []

    # layer 0 query nodes: [3B]
    nodes_l = torch.cat([root_src, root_dst, root_neg], dim=0)
    ts_l    = torch.cat([root_ts,  root_ts,  root_ts],  dim=0)

    for l in range(num_layers):
        nvtx.range_push(f"sampling layer {l}")
        layer = _fixed_sample_one_layer(
            g=g,
            sampler=sampler,
            nodes=nodes_l,
            times=ts_l,
            nfeat=nfeat,
            efeat=efeat,
            layer_idx=l,
            num_neighbors=num_neighbors,
        )
        nvtx.range_pop()  # sampling layer {l}
        atlas_layers.append(layer)

        # next-layer query = flattened src_neighbors from current layer
        nodes_l = layer.src_nodes.reshape(-1)
        ts_l    = layer.src_ts.reshape(-1)

    nvtx.range_pop()  # multi-layer sampling

    return AtlasCPUBlock(
        batch_size=B,
        num_layers=num_layers,
        num_neighbors=num_neighbors,
        root_src=_pin(root_src),
        root_dst=_pin(root_dst),
        root_neg=_pin(root_neg),
        root_ts=_pin(root_ts),
        root_src_feat=_pin(root_src_feat),
        root_dst_feat=_pin(root_dst_feat),
        root_neg_feat=_pin(root_neg_feat),
        root_edge_feat=_pin(root_edge_feat),
        atlas_layers=atlas_layers,
    )


def _fixed_sample_one_layer_index(
    g,
    sampler: FixedTSampler,
    nodes: torch.Tensor,         # [N_l]
    times: torch.Tensor,         # [N_l]
    layer_idx: int,
    num_neighbors: int,
    unique_node_chunks: List[torch.Tensor],
    unique_edge_chunks: List[torch.Tensor],
) -> AtlasLayer:
    """
    One-layer sampling -> AtlasLayer (flat view)
    """
    device = torch.device("cpu")
    nvtx.range_push("fixed sample one layer")

    nodes = nodes.to(device, dtype=torch.long)
    times = times.to(device, dtype=torch.float32)

    N_l = nodes.shape[0]
    K = num_neighbors

    # --------- call C++ sampler ----------
    nodes_np = nodes.to(dtype=torch.int32).cpu().numpy()
    times_np = times.cpu().numpy()  # already float32

    pad_mask = (nodes_np < 0)        # padding nodes do not participate in real sampling
    nodes_q = nodes_np.copy()
    times_q = times_np.copy()
    nodes_q[pad_mask] = 0
    times_q[pad_mask] = 0.0

    block = sampler.sample_block(g, nodes_q, times_q)

    dstindex_np = block.copy_dstindex()   # [N_l * K]
    srcnodes_np = block.copy_srcnodes()   # [N_l * K]
    eids_np     = block.copy_eid()
    ets_np      = block.copy_ets()

    nvtx.range_pop()
    assert dstindex_np.shape[0] == N_l * K, "fixed sampler should output N_l*K edges"

    nvtx.range_push("padding sampled data")
    # --------- fill padding positions with -1 ----------
    if pad_mask.any():
        idx = np.where(pad_mask)[0]
        src_pad = srcnodes_np.reshape(N_l, K)
        eids_pad = eids_np.reshape(N_l, K)
        ets_pad  = ets_np.reshape(N_l, K)
        src_pad[idx, :] = -1
        eids_pad[idx, :] = -1
        ets_pad[idx, :] = 0.0
        srcnodes_np = src_pad.reshape(-1)
        eids_np     = eids_pad.reshape(-1)
        ets_np      = ets_pad.reshape(-1)

    srcnodes_tensor = torch.from_numpy(srcnodes_np).to(device=device, dtype=torch.long)
    eids_tensor = torch.from_numpy(eids_np).to(device=device, dtype=torch.long)

    unique_node_chunks.append(nodes)
    unique_node_chunks.append(srcnodes_tensor.reshape(-1))
    unique_edge_chunks.append(eids_tensor.reshape(-1))

    nvtx.range_pop()
    nvtx.range_push("convert to torch & reshape")
    nvtx.range_push("pin dst nodes & times")
    dst_nodes = _pin(nodes)        # [N_l]
    dst_ts    = _pin(times)        # [N_l]
    nvtx.range_push("pin dst feat")
    # dst_feat  = _pin(nfeat[dst_nodes.clamp(min=0)])   # treat padding (-1) as node 0 for safe gather
    # dst_feat = pinned_gather(nfeat, dst_nodes.clamp(min=0)).view(N_l, -1)
    nvtx.range_pop()
    nvtx.range_pop()
    nvtx.range_push("pin src nodes, times & edge ids")
    src_nodes = _pin(srcnodes_tensor.view(N_l, K))
    src_ts    = _pin(torch.from_numpy(ets_np).to(device=device, dtype=torch.float32).view(N_l, K))
    edge_ids  = _pin(eids_tensor.view(N_l, K))

    src_nodes_clamp = src_nodes
    src_nodes_clamp[src_nodes_clamp < 0] = 0
    nvtx.range_push("pin src feat")
    # src_feat = _pin(nfeat[src_nodes_clamp])       # [N_l, K, Dn]
    # src_feat = pinned_gather(nfeat, src_nodes_clamp.reshape(-1)).view(N_l, K, -1)
    nvtx.range_pop()
    nvtx.range_push("clamp edge ids & pin edge feat")
    edge_ids_clamp = edge_ids
    edge_ids_clamp[edge_ids_clamp < 0] = 0
    nvtx.range_pop()
    nvtx.range_push("pin edge feat")
    # edge_feat = _pin(efeat[edge_ids_clamp])       # [N_l, K, De]
    # edge_feat = pinned_gather(efeat, edge_ids_clamp.reshape(-1)).view(N_l, K, -1)
    nvtx.range_pop()
    nvtx.range_pop()


    nvtx.range_pop()
    return AtlasLayer(
        layer=layer_idx,
        dst_nodes=dst_nodes,
        dst_ts=dst_ts,
        dst_feat=None,
        src_nodes=src_nodes,
        src_ts=src_ts,
        src_feat=None,
        edge_ids=edge_ids,
        edge_feat=None,
    ), unique_node_chunks, unique_edge_chunks



def make_batch_index(
    g: "TGraph",
    start: int,
    end: int,
    neg_sampler: Callable[[int], np.ndarray],
    sampler: FixedTSampler,
    num_layers: int,
    num_neighbors: int,
) -> AtlasCPUBlock:
    """
    Build AtlasCPUBlock from edges in [start, end) with sampler
    """
    device = torch.device("cpu")

    nvtx.range_push("make edges")
    # ---- fetch current batch edges ----
    edges = g._edges[start:end]     # [B, 2]
    ts    = g._times[start:end]        # [B]

    if isinstance(edges, np.ndarray):
        edges = torch.from_numpy(edges)
    if isinstance(ts, np.ndarray):
        ts = torch.from_numpy(ts)

    B = edges.shape[0]
    src = edges[:, 0].to(device, dtype=torch.long)
    dst = edges[:, 1].to(device, dtype=torch.long)
    ts  = ts.to(device, dtype=torch.float32)

    # ---- root triples ----
    root_src = src
    root_dst = dst
    root_ts  = ts

    neg_np   = neg_sampler(B).astype(np.int64)
    root_neg = torch.from_numpy(neg_np).to(device=device, dtype=torch.long)
    nvtx.range_pop()  # make edges

    nvtx.range_push("make features")
    # ---- feature tables ----
    nfeat = g.nfeat
    efeat = g.efeat
    if isinstance(nfeat, np.ndarray):
        nfeat = torch.from_numpy(nfeat)
    if isinstance(efeat, np.ndarray):
        efeat = torch.from_numpy(efeat)
    nfeat = nfeat.to(device)
    efeat = efeat.to(device)

    Dn = nfeat.shape[1]
    De = efeat.shape[1]

    eid_pos = torch.arange(start, end, dtype=torch.long, device=device)

    root_src_feat = nfeat[root_src]
    root_dst_feat = nfeat[root_dst]
    root_neg_feat = nfeat[root_neg]
    root_edge_feat = efeat[eid_pos]
    nvtx.range_pop()  # make features

    nvtx.range_push("multi-layer sampling")
    # ---- multi-layer sampling ----
    atlas_layers: List[AtlasLayer] = []

    # layer 0 query nodes: [3B]
    nodes_l = torch.cat([root_src, root_dst, root_neg], dim=0)
    ts_l    = torch.cat([root_ts,  root_ts,  root_ts],  dim=0)

    unique_node_chunks: List[torch.Tensor] = []
    unique_edge_chunks: List[torch.Tensor] = []

    for l in range(num_layers):
        nvtx.range_push(f"sampling layer {l}")
        layer, unique_node_chunks, unique_edge_chunks = _fixed_sample_one_layer_index(
            g=g,
            sampler=sampler,
            nodes=nodes_l,
            times=ts_l,
            layer_idx=l,
            num_neighbors=num_neighbors,
            unique_node_chunks=unique_node_chunks,
            unique_edge_chunks=unique_edge_chunks,
        )
        nvtx.range_pop()  # sampling layer {l}
        atlas_layers.append(layer)

        # next-layer query = flattened src_neighbors from current layer
        nodes_l = layer.src_nodes.reshape(-1)
        ts_l    = layer.src_ts.reshape(-1)

    nvtx.range_pop()  # multi-layer sampling

    nvtx.range_push("process unique nodes & edges")
    if unique_node_chunks:
        unique_nodes = torch.unique(torch.cat(unique_node_chunks, dim=0), sorted=True)
    else:
        unique_nodes = torch.empty(0, dtype=torch.long)
    if unique_edge_chunks:
        unique_edges = torch.unique(torch.cat(unique_edge_chunks, dim=0), sorted=True)
    else:
        unique_edges = torch.empty(0, dtype=torch.long)
    nvtx.range_pop()

    nvtx.range_push("gather feature pools")

    node_mask = unique_nodes >= 0
    edge_mask = unique_edges >= 0

    valid_nodes = unique_nodes[node_mask]
    valid_edges = unique_edges[edge_mask]

    if valid_nodes.numel() == 0:
        nodes_feat_pool = torch.empty((0, Dn), dtype=nfeat.dtype, device=nfeat.device)
    else:
        nodes_feat_pool = pinned_gather(nfeat, valid_nodes)

    if valid_edges.numel() == 0:
        edges_feat_pool = torch.empty((0, De), dtype=efeat.dtype, device=efeat.device)
    else:
        edges_feat_pool = pinned_gather(efeat, valid_edges)

    nvtx.range_push("pin feature pools")
    nodes_feat_pool = _pin(nodes_feat_pool)
    nvtx.range_pop()
    nvtx.range_push("pin feature pools edges")
    edges_feat_pool = _pin(edges_feat_pool)
    nvtx.range_pop()

    unique_nodes = valid_nodes
    unique_edges = valid_edges
    nvtx.range_pop()
    return AtlasCPUBlock(
        batch_size=B,
        num_layers=num_layers,
        num_neighbors=num_neighbors,
        root_src=_pin(root_src),
        root_dst=_pin(root_dst),
        root_neg=_pin(root_neg),
        root_ts=_pin(root_ts),
        root_src_feat=_pin(root_src_feat),
        root_dst_feat=_pin(root_dst_feat),
        root_neg_feat=_pin(root_neg_feat),
        root_edge_feat=_pin(root_edge_feat),
        atlas_layers=atlas_layers,
        unique_nodes=_pin(unique_nodes),
        unique_edges=_pin(unique_edges),
        nodes_feat_pool=nodes_feat_pool,
        edges_feat_pool=edges_feat_pool
    )
