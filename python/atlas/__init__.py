"""
atlas: Asynchronous Temporal Latency-Aware framework for temporal GNNs.

Top-level API:

- Core data structures: TGraph, TBatch, TBlock, TFrame
- States & context:    Memory, Mailbox, TContext
- Sampling & ops:      TSampler, nn, op
- Utilities:           utils, iter_edges, EdgesIter
"""

from __future__ import annotations

__version__ = "0.0.1"


from ._graph import TGraph, from_csv
# from ._batch import TBatch
# from ._block import TBlock
from ._frame import TFrame
from ._memory import Memory
from ._mailbox import Mailbox
from ._sampler import TSampler, FixedTSampler
# from ._context import TContext
from ._core import TError
from . import _utils as utils
# from . import nn
# from . import op
from ._cpublock import AtlasCPUBlock, AtlasLayer, _pin
from ._gpublock import AtlasGPUBlock, AtlasGPULayer, cpu_to_gpu_block, recover_block
from . import nn_layers
from ._utils import make_batch, pinned_gather, make_batch_index
from ._stats import tt

__all__ = [
    "__version__",
    "TGraph",
    "from_csv",
    "TBatch",
    "TBlock",
    "TFrame",
    "Memory",
    "Mailbox",
    "TSampler",
    "FixedTSampler",
    "TContext",
    "TError",
    "utils",
    "nn",
    "op",
    "iter_edges",
    "EdgesIter",
]



def iter_edges(
    g: TGraph,
    size: int = 1,
    start: int | None = None,
    end: int | None = None,
) -> "EdgesIter":
    """
    Create an iterator that yields TBatch objects over the edges of a TGraph.

    Parameters
    ----------
    g : TGraph
        The graph to iterate on.
    size : int, default=1
        Number of edges in each mini-batch.
    start : int or None, default=None
        Starting edge index (inclusive). If None, starts from 0.
    end : int or None, default=None
        Ending edge index (exclusive). If None, iterates to the last edge.

    Returns
    -------
    EdgesIter
        An iterator that yields TBatch objects.
    """
    return EdgesIter(g, size=size, start=start, end=end)


class EdgesIter:
    """An edge iterator over a TGraph that yields TBatch objects."""

    def __init__(
        self,
        g: TGraph,
        size: int = 1,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        self._g = g
        self._size = size
        self._curr = 0 if start is None else start
        self._last = g.num_edges() if end is None else end

    def __iter__(self) -> "EdgesIter":
        return self

    def __next__(self) -> TBatch:
        if self._curr < self._last:
            idx = self._curr
            self._curr += self._size
            end = min(self._curr, self._last)
            return TBatch(self._g, range=(idx, end))
        raise StopIteration
