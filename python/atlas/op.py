import torch
import numpy as np
import torch_scatter
from torch import Tensor
from typing import Any, Callable, List, Union

from . import _c
from ._core import TError
from ._block import TBlock
# from ._context import TContext
# from ._stats import tt

from torch.cuda import nvtx


# def find_last_message(uniq_nodes: np.ndarray, sorted_edges: np.ndarray):
#     msg_order, msg_index = _c.find_last_message(uniq_nodes, sorted_edges)
#     msg_order = msg_order.reshape(-1, 2)
#     return msg_order, msg_index


def edge_view(blk: TBlock, data: Tensor) -> Tensor:
    '''
    Reindex the data of edges based on the target node indices in the TBlock.
    
    :param blk:
    :param data:
    '''
    blk._check_has_nbrs()
    assert data.shape[0] == blk._dstdata.dim()
    idx = torch.from_numpy(blk._dstindex)
    idx = idx.to(device=data.device, dtype=torch.long)
    return data[idx]


def edge_softmax(blk: TBlock, data: Tensor) -> Tensor:
    '''
    Computes segmented softmax on given data using edge information from the block.
    
    :param blk:
    :param data:
    '''
    blk._check_has_nbrs()
    size = blk._edata.dim()
    assert data.shape[0] == size
    reindex = torch.from_numpy(blk._dstindex)
    reindex = torch.unique(reindex, return_inverse=True)[1]
    reindex = reindex.to(device=data.device, dtype=torch.long)
    return torch_scatter.scatter_softmax(data, reindex, dim=0, dim_size=size)


def edge_reduce(blk: TBlock, data: Tensor, op='sum') -> Tensor:
    '''
    Computes segmented reduction (e.g. sum or mean) on given data 
    using edge information from the block.
    
    :param blk:
    :param data:
    '''
    blk._check_has_nbrs()
    assert op in ['sum', 'mean'], "currently only supports sum or mean"
    assert data.shape[0] == blk._edata.dim()
    size = blk._dstdata.dim()
    scatter_idx = torch.from_numpy(blk._dstindex)
    scatter_idx = scatter_idx.to(device=data.device, dtype=torch.long)
    return torch_scatter.segment_coo(data, scatter_idx, dim_size=size, reduce=op)


def src_scatter(blk: TBlock, data: Tensor, op='sum') -> Tensor:
    '''
    Aggregates the features of the source node indices in TBlock, 
    using the specified aggregation operation ('sum' or 'mean').
    
    :param blk:
    :param data:
    :param op:
    '''
    blk._check_has_nbrs()
    assert op in ['sum', 'mean'], "currently only supports sum or mean"
    assert data.shape[0] == len(blk._srcnodes)
    uniq_nids, idx = blk.uniq_src()
    idx = idx.to(data.device)
    return torch_scatter.scatter(data, idx, dim=0, dim_size=len(uniq_nids), reduce=op)


def coalesce(blk: TBlock, by='latest') -> TBlock:
    '''
    Segmented operation to reduce source nodes for each destination node 
    by a certain property, such as latest timestamp.
    
    :param blk:
    :param by: method to sample source nodes
    '''
    assert by == 'latest', "currently only supports latest"
    assert blk.has_nbrs() and len(blk.dstnodes) == len(blk.srcnodes)
    uniq_nodes, uniq_idx = np.unique(blk.dstnodes, return_index=True)
    idx = _c.find_latest_uniq(uniq_nodes, blk.dstnodes, blk.ets)
    src = blk.srcnodes[idx]
    eid = blk.eid[idx]
    ets = blk.ets[idx]
    blk._replace_dst(uniq_nodes, blk.dsttimes[uniq_idx])
    blk.set_nbrs(np.arange(len(src)), src, eid, ets)
    return blk


def preload(blk: TBlock, use_pin=True):
    '''
    Prefetch data (e.g. features, memory, mails) needed by the TBlock 
    and its subsequent blocks for computations.

    :param blk:
    :param use_pin: whether to pin memory
    '''
    curr = blk
    while curr.next is not None:
        curr = curr.next
    while curr is not None:
        if curr.num_dst() > 0:
            if curr.next is None:
                curr._load_mail(use_pin=use_pin)
                curr._load_mem_data(use_pin=use_pin)
            if curr.has_nbrs():
                if curr.next is None:
                    curr._load_nfeat(use_pin=use_pin)
                curr._load_efeat(use_pin=use_pin)
        curr = curr.prev


def aggregate(blk: TBlock, fn_or_list: Union[Callable, List[Callable]], key: str = None) -> Any:
    '''
    Performs pull-style multi-hop aggregation from the tail block 
    back towards the given block by applying function to each block, 
    using the key to pass along results.
    
    :param blk:
    :param fn_or_list:
    :param key:
    '''
    while blk.next is not None:
        blk = blk.next
    output = None
    while blk is not None:
        if blk.num_dst() == 0:
            output = blk.run_hooks(output)
        elif isinstance(fn_or_list, List):
            output = blk.apply(fn_or_list[blk.layer])
        else:
            output = blk.apply(fn_or_list)
        # t_start = tt.start()
        nvtx.range_push("prepare input block")
        if blk.prev is not None and output is not None and key:
            if blk._include_prev_dst:
                num_dst = blk.prev.num_dst()
                blk.prev.dstdata[key] = output[:num_dst]
                blk.prev.srcdata[key] = output[num_dst:]
            else:
                blk.prev.srcdata[key] = output
        blk.clear_data()
        blk.clear_hooks()
        # 
        # tt.t_prep_input += tt.elapsed(t_start)
        nvtx.range_pop()  # prepare input block
        blk = blk.prev
    return output


def propagate(blk: TBlock, fn_or_list: Union[Callable, List[Callable]]) -> Any:
    '''
    Performs push-style multi-hop propagation from the given block 
    to the tail block by applying function to each block.
    
    :param blk:
    :param fn_or_list:
    '''
    output = None
    while blk is not None:
        if blk.num_dst() == 0:
            output = blk.run_hooks(output)
        elif isinstance(fn_or_list, List):
            output = blk.apply(fn_or_list[blk.layer])
        else:
            output = blk.apply(fn_or_list)
        blk.clear_data()
        blk.clear_hooks()
        blk = blk.next
    return output
