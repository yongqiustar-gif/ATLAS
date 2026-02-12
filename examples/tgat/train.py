import argparse
import os
import numpy as np
import torch
import atlas as at

from tgat import TGAT
import support

### arguments

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True, help='dataset name')
parser.add_argument('--data-path', type=str, default='', help='path to data folder')
parser.add_argument('--prefix', type=str, default='', help='name for saving trained model')
parser.add_argument('--gpu', type=int, default=0, help='gpu device to use (or -1 for cpu)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 100)')
parser.add_argument('--bsize', type=int, default=200, help='batch size (default: 200)')
parser.add_argument('--lr', type=str, default=0.0001, help='learning rate (default: 1e-4)')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate (default: 0.1)')
parser.add_argument('--n-layers', type=int, default=2, help='number of layers (default: 2)')
parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads (default: 2)')
parser.add_argument('--n-nbrs', type=int, default=20, help='number of neighbors to sample (default: 20)')
parser.add_argument('--dim-time', type=int, default=100, help='dimension of time features (default: 100)')
parser.add_argument('--dim-embed', type=int, default=100, help='dimension of embeddings (default: 100)')
parser.add_argument('--seed', type=int, default=-1, help='random seed to use')
parser.add_argument('--move', action='store_true', help='move data to device')
parser.add_argument('--n-threads', type=int, default=32, help='number of threads for sampler (default: 32)')
parser.add_argument('--sampling', type=str, default='recent', choices=['recent', 'uniform'], help='sampling strategy (default: recent)')
args = parser.parse_args()
print(args)

device = support.make_device(args.gpu)
model_path = support.make_model_path('tgat', args.prefix, args.data)
if args.seed >= 0:
    support.set_seed(args.seed)

DATA: str = args.data
DATA_PATH: str = args.data_path
EPOCHS: int = args.epochs
BATCH_SIZE: int = args.bsize
LEARN_RATE: float = float(args.lr)
DROPOUT: float = float(args.dropout)
N_LAYERS: int = args.n_layers
N_HEADS: int = args.n_heads
N_NBRS: int = args.n_nbrs
DIM_TIME: int = args.dim_time
DIM_EMBED: int = args.dim_embed
N_THREADS: int = args.n_threads
SAMPLING: str = args.sampling

### load data

g = support.load_graph(os.path.join(DATA_PATH, f'data/{DATA}/edges.csv'))
support.load_feats(g, DATA, DATA_PATH)
dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
dim_nfeat = g.nfeat.shape[1]

g.set_compute(device)
if args.move:
    g.move_data(device)

g._get_tcsr()


### model

sampler = at.TSampler(N_NBRS, strategy=SAMPLING, num_threads=N_THREADS)
model = TGAT(g,
    dim_node=dim_nfeat,
    dim_edge=dim_efeat,
    dim_time=DIM_TIME,
    dim_embed=DIM_EMBED,
    sampler=sampler,
    num_layers=N_LAYERS,
    num_heads=N_HEADS,
    dropout=DROPOUT)
model = model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, capturable=True)


### training

train_end, val_end = support.data_split(g.num_edges(), 0.7, 0.15)
neg_sampler = lambda size: np.random.randint(0, g.num_nodes(), size)

trainer = support.LinkPredTrainer(
    g, model, criterion, optimizer, neg_sampler,
    EPOCHS, BATCH_SIZE, train_end, val_end,
    model_path, None)

trainer.train()
trainer.test()