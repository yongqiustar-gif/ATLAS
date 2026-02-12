import random
import sys
import time
import os
import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Callable, Optional, Tuple, Union

import atlas as at

from torch.cuda import nvtx


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_device(gpu: int) -> torch.device:
    return torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')


def make_model_path(model: str, prefix: str, data: str) -> str:
    """If prefix is not empty, return 'models/{model}/{prefix}-{data}.pt', else return
    'models/{model}/{data}-{time.time()}.pt'."""
    Path(f'models/{model}').mkdir(parents=True, exist_ok=True)
    if prefix:
        return f'models/{model}/{prefix}-{data}.pt'
    else:
        return f'models/{model}/{data}-{time.time()}.pt'


def make_model_mem_path(model: str, prefix: str, data: str) -> str:
    Path(f'models/{model}').mkdir(parents=True, exist_ok=True)
    if prefix:
        return f'models/{model}/{prefix}-{data}-mem.pt'
    else:
        return f'models/{model}/{data}-mem-{time.time()}.pt'
    

def load_graph(path: Union[str, Path]) -> at.TGraph:
    """Create a TGraph with edges and timestamps loaded from path. Provided data should include
    'src' 'dst' and 'time' columns."""
    df = pd.read_csv(str(path))

    src = df['src'].to_numpy().astype(np.int32).reshape(-1, 1)
    dst = df['dst'].to_numpy().astype(np.int32).reshape(-1, 1)
    etime = df['time'].to_numpy().astype(np.float32)
    del df

    edges = np.concatenate([src, dst], axis=1)
    del src
    del dst

    g = at.TGraph(edges, etime)
    print('num edges:', g.num_edges())
    print('num nodes:', g.num_nodes())
    return g


def load_feats(g: at.TGraph, d: str, data_path: str=''):
    """
    Load edge features and node features to g from data/{d}/edge_features.pt and
    data/{d}/edge_features.pt. If no file, create random edge and node features for data 'mooc',
    'lastfm' and 'wiki-talk', create random edge features for data 'wiki' and 'reddit', None for
    other data.
    """
    edge_feats = None
    node_feats = None

    if Path(os.path.join(data_path, f'data/{d}/edge_features.pt')).exists():
        edge_feats = torch.load(os.path.join(data_path, f'data/{d}/edge_features.pt'))
        edge_feats = edge_feats.type(torch.float32)
    elif d in ['mooc', 'lastfm', 'wiki-talk']:
        edge_feats = torch.randn(g.num_edges(), 128, dtype=torch.float32)

    if Path(os.path.join(data_path, f'data/{d}/node_features.pt')).exists():
        node_feats = torch.load(os.path.join(data_path, f'data/{d}/node_features.pt'))
        node_feats = node_feats.type(torch.float32)
    elif d in ['wiki', 'mooc', 'reddit', 'lastfm', 'wiki-talk']:
        node_feats = torch.randn(g.num_nodes(), edge_feats.shape[1], dtype=torch.float32)

    print('edge feat:', None if edge_feats is None else edge_feats.shape)
    print('node feat:', None if node_feats is None else node_feats.shape)
    g.efeat = edge_feats
    g.nfeat = node_feats


def data_split(num_samples: int, train_percent: float, val_percent: float) -> Tuple[int, int]:
    train_end = int(np.ceil(num_samples * train_percent))
    val_end = int(np.ceil(num_samples * (train_percent + val_percent)))
    return train_end, val_end


class EdgePredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.src_fc = nn.Linear(dim, dim)
        self.dst_fc = nn.Linear(dim, dim)
        self.out_fc = nn.Linear(dim, 1)
        self.act = nn.ReLU()

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
        h_src = self.src_fc(src)
        h_dst = self.dst_fc(dst)
        h_out = self.act(h_src + h_dst)
        return self.out_fc(h_out)
    

