from . import _c
from ._core import TError
from ._block import TBlock
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._graph import TGraph
from ._utils import get_num_cpus
# from ._stats import tt
from torch.cuda import nvtx
import numpy as np

class TSampler(object):

    def __init__(self, num_nbrs: int, strategy='recent', num_threads: int = None):
        """
        Internal constructor for creating a TSampler

        :param int num_nbrs: number of neighbors
        :param str strategy: sampling strategy, 'recent' or 'uniform'
        :param int num_threads: number of threads for parallel sampling, set to number of cpus if not provided
        :raises TError: if strategy is not in ['recent', 'uniform']
        """

        if strategy not in ['recent', 'uniform']:
            raise TError(f'sampling strategy not supported: {strategy}')

        self._n_nbrs = num_nbrs
        self._strategy = strategy
        self._n_threads = get_num_cpus() \
            if num_threads is None else num_threads

        self._sampler = _c.TemporalSampler(
            self._n_threads,
            self._n_nbrs,
            self._strategy == 'recent')

    def sample(self, blk: TBlock) -> TBlock:
        """Updates block with sampled 1-hop source neighbors
        
        :returns: updated block
        """
        # t_start = tt.start()
        nvtx.range_push("neighbor sampling")
        if blk.num_dst() > 0:
            block = self._sampler.sample(blk._g._get_tcsr(), blk._dstnodes, blk._dsttimes)
            blk.set_nbrs(
                block.copy_dstindex(),
                block.copy_srcnodes(),
                block.copy_eid(),
                block.copy_ets())
        # tt.t_sample += tt.elapsed(t_start)
        nvtx.range_pop()  # sampling
        return blk


class FixedTSampler(TSampler):

    def __init__(self, num_nbrs: int, strategy='recent', num_threads: int = None):
        super().__init__(num_nbrs, strategy=strategy, num_threads=num_threads)
        self._sampler = _c.FixedTemporalSampler(
            self._n_threads,
            self._n_nbrs,
            self._strategy == 'recent')
        
    def sample_block(
            self,
            g: "TGraph",
            nodes: np.ndarray,
            times: np.ndarray,
        ) -> "_c.TemporalBlock":
        """Directly call the C++ fixed sampler with NumPy buffers."""

        tcsr = g._get_tcsr()
        if nodes.dtype != np.int32:
            nodes = nodes.astype(np.int32, copy=False)
        if times.dtype != np.float32:
            times = times.astype(np.float32, copy=False)
        return self._sampler.sample(tcsr, nodes, times)
