import argparse
import os
from pathlib import Path

import numpy as np
import torch
import atlas as at

torch.set_float32_matmul_precision('high')

from tgat import AtlasTGAT
import support
from async_trainer import AsyncPipelineTrainer, configure_cpu_threads, AsyncPipelineTrainerIndex


parser = argparse.ArgumentParser(description="Async TGAT trainer with CUDA Graph support")
parser.add_argument('-d', '--data', type=str, required=True, help='dataset name')
parser.add_argument('--data-path', type=str, default='', help='path to data folder')
parser.add_argument('--prefix', type=str, default='', help='name for saving trained model')
parser.add_argument('--gpu', type=int, default=0, help='gpu device to use (or -1 for cpu)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 100)')
parser.add_argument('--bsize', type=int, default=200, help='batch size (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate (default: 0.1)')
parser.add_argument('--n-layers', type=int, default=2, help='number of layers (default: 2)')
parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads (default: 2)')
parser.add_argument('--n-nbrs', type=int, default=20, help='number of neighbors to sample (default: 20)')
parser.add_argument('--dim-time', type=int, default=100, help='dimension of time features (default: 100)')
parser.add_argument('--dim-embed', type=int, default=100, help='dimension of embeddings (default: 100)')
parser.add_argument('--seed', type=int, default=-1, help='random seed to use')
parser.add_argument('--n-threads', type=int, default=64, help='number of threads for sampler (default: 64)')
parser.add_argument('--sampling', type=str, default='recent', choices=['recent', 'uniform'], help='sampling strategy (default: recent)')
parser.add_argument('--cpu-queue-size', type=int, default=8, help='prefetch queue size for CPU make_batch stage')
parser.add_argument('--gpu-queue-size', type=int, default=4, help='GPU prefetch depth (ready batches kept on device)')
parser.add_argument('--h2d-streams', type=int, default=1, help='number of CUDA streams for H2D overlap (only when gpu>=0)')
parser.add_argument('--disable-cudagraph', action='store_true', help='disable CUDA Graph even when running on GPU')
parser.add_argument('--torch-compile', action='store_true', help='wrap model with torch.compile for ahead-of-time graph optimizations')
args = parser.parse_args()
print(args)


device = support.make_device(args.gpu)
model_path = support.make_model_path('tgat-async', args.prefix, args.data)
model_mem_path = support.make_model_mem_path('tgat-async', args.prefix, args.data)
if args.seed >= 0:
    support.set_seed(args.seed)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent
DATA_ROOT = Path(args.data_path).expanduser().resolve() if args.data_path else DEFAULT_DATA_ROOT

N_THREADS = configure_cpu_threads(args.n_threads)
print(f"using {N_THREADS} CPU threads (logical cores={os.cpu_count()})")
print(f"CPU queue={max(1, args.cpu_queue_size)}, GPU prefetch={max(1, args.gpu_queue_size)}, H2D streams={args.h2d_streams if device.type == 'cuda' else 0}")

g = support.load_graph(DATA_ROOT / 'data' / args.data / 'edges.csv')
support.load_feats(g, args.data, str(DATA_ROOT))
dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
dim_nfeat = g.nfeat.shape[1]

g._get_tcsr()

sampler = at.FixedTSampler(args.n_nbrs, strategy=args.sampling, num_threads=N_THREADS)
model = AtlasTGAT(
    dim_node=dim_nfeat,
    dim_edge=dim_efeat,
    dim_time=args.dim_time,
    dim_embed=args.dim_embed,
    num_layers=args.n_layers,
    num_heads=args.n_heads,
    dropout=args.dropout,
).to(device)
if args.torch_compile:
    compile_mode = "max-autotune"
    try:
        model = torch.compile(model, mode=compile_mode)
        print(f"torch.compile enabled (mode={compile_mode})")
    except Exception as exc:
        print(f"[warn] torch.compile failed: {exc}; continuing without compilation")
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, capturable=(device.type == 'cuda'))

train_end, val_end = support.data_split(g.num_edges(), 0.7, 0.15)
neg_sampler = lambda size: np.random.randint(0, g.num_nodes(), size)

trainer = AsyncPipelineTrainerIndex(
    g=g,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    sampler=sampler,
    neg_sampler=neg_sampler,
    epochs=args.epochs,
    batch_size=args.bsize,
    train_end=train_end,
    val_end=val_end,
    num_layers=args.n_layers,
    num_neighbors=args.n_nbrs,
    device=device,
    model_path=model_path,
    model_mem_path=model_mem_path,
    cpu_queue_size=args.cpu_queue_size,
    gpu_queue_size=args.gpu_queue_size,
    h2d_streams=args.h2d_streams,
    use_cuda_graph=not args.disable_cudagraph,
)

trainer.train()
trainer.test()
