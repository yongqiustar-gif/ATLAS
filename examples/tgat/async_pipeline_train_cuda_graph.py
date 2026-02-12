import argparse
import queue
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import atlas as at

import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

from tgat import AtlasTGAT
import support

from torch.cuda import nvtx

from atlas import tt

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
parser.add_argument('--n-threads', type=int, default=64, help='number of threads for sampler (default: 32)')
parser.add_argument('--sampling', type=str, default='recent', choices=['recent', 'uniform'], help='sampling strategy (default: recent)')
parser.add_argument('--cpu-queue-size', type=int, default=8, help='prefetch queue size for CPU make_batch stage')
parser.add_argument('--gpu-queue-size', type=int, default=4, help='GPU prefetch depth (ready batches kept on device)')
parser.add_argument('--h2d-streams', type=int, default=1, help='number of CUDA streams for H2D overlap (only when gpu>=0)')
args = parser.parse_args()
print(args)


def _configure_cpu_threads(requested: int) -> int:
    logical = os.cpu_count() or 16
    target = logical if requested <= 0 else min(requested, logical)
    target = max(1, target)
    env_val = str(target)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, env_val)
    torch.set_num_threads(target)
    torch.set_num_interop_threads(max(1, min(4, target // 4 or 1)))
    return target


device = support.make_device(args.gpu)
model_path = support.make_model_path('tgat-async', args.prefix, args.data)
model_mem_path = support.make_model_mem_path('tgat-async', args.prefix, args.data)
if args.seed >= 0:
    support.set_seed(args.seed)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent
DATA_ROOT = Path(args.data_path).expanduser().resolve() if args.data_path else DEFAULT_DATA_ROOT

DATA: str = args.data
EPOCHS: int = args.epochs
BATCH_SIZE: int = args.bsize
LEARN_RATE: float = float(args.lr)
DROPOUT: float = float(args.dropout)
N_LAYERS: int = args.n_layers
N_HEADS: int = args.n_heads
N_NBRS: int = args.n_nbrs
DIM_TIME: int = args.dim_time
DIM_EMBED: int = args.dim_embed
N_THREADS: int = _configure_cpu_threads(args.n_threads)
SAMPLING: str = args.sampling
CPU_QUEUE_SIZE: int = max(1, args.cpu_queue_size)
GPU_QUEUE_SIZE: int = max(1, args.gpu_queue_size)
H2D_STREAMS: int = max(1, args.h2d_streams)
NVTX_ENABLED: bool = torch.cuda.is_available() and device.type == 'cuda'

print(f"using {N_THREADS} CPU threads (logical cores={os.cpu_count()})")
print(f"CPU queue={CPU_QUEUE_SIZE}, GPU prefetch={GPU_QUEUE_SIZE}, H2D streams={H2D_STREAMS if device.type == 'cuda' else 0}")

edge_csv = DATA_ROOT / "data" / DATA / "edges.csv"
g = support.load_graph(edge_csv)
support.load_feats(g, DATA, str(DATA_ROOT))
dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
dim_nfeat = g.nfeat.shape[1]

g._get_tcsr()

sampler = at.FixedTSampler(N_NBRS, strategy=SAMPLING, num_threads=N_THREADS)
model = AtlasTGAT(
    dim_node=dim_nfeat,
    dim_edge=dim_efeat,
    dim_time=DIM_TIME,
    dim_embed=DIM_EMBED,
    num_layers=N_LAYERS,
    num_heads=N_HEADS,
    dropout=DROPOUT,
).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, capturable=True)


def clone_static_gpu_block(template: "at.AtlasGPUBlock", device: torch.device) -> "at.AtlasGPUBlock":
    """
    Allocate an empty static block on the same device from a sample AtlasGPUBlock.
    Same structure; tensors are empty buffers used as fixed CUDA Graph inputs.
    """
    def empty_like(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x, device=device)

    static_layers: list[at.AtlasGPULayer] = []
    for lyr in template.layers:
        static_layers.append(
            at.AtlasGPULayer(
                layer=lyr.layer,
                dst_nodes=empty_like(lyr.dst_nodes),
                dst_ts=empty_like(lyr.dst_ts),
                dst_feat=empty_like(lyr.dst_feat),
                src_nodes=empty_like(lyr.src_nodes),
                src_ts=empty_like(lyr.src_ts),
                src_feat=empty_like(lyr.src_feat),
                edge_ids=empty_like(lyr.edge_ids),
                edge_feat=empty_like(lyr.edge_feat),
            )
        )

    return at.AtlasGPUBlock(
        batch_size=template.batch_size,
        num_layers=template.num_layers,
        num_neighbors=template.num_neighbors,
        root_src=empty_like(template.root_src),
        root_dst=empty_like(template.root_dst),
        root_neg=empty_like(template.root_neg),
        root_ts=empty_like(template.root_ts),
        root_src_feat=empty_like(template.root_src_feat),
        root_dst_feat=empty_like(template.root_dst_feat),
        root_neg_feat=empty_like(template.root_neg_feat),
        root_edge_feat=empty_like(template.root_edge_feat),
        layers=static_layers,
    )


def copy_gpu_block_into_static(src: "at.AtlasGPUBlock", dst: "at.AtlasGPUBlock") -> None:
    """
    Copy data from dynamic gpu_block into tensors inside static_block.
    Assumes both sides have exactly the same shapes.
    """
    # root
    dst.root_src.copy_(src.root_src)
    dst.root_dst.copy_(src.root_dst)
    dst.root_neg.copy_(src.root_neg)
    dst.root_ts.copy_(src.root_ts)
    dst.root_src_feat.copy_(src.root_src_feat)
    dst.root_dst_feat.copy_(src.root_dst_feat)
    dst.root_neg_feat.copy_(src.root_neg_feat)
    dst.root_edge_feat.copy_(src.root_edge_feat)

    # layers
    assert len(dst.layers) == len(src.layers)
    for sl, dl in zip(src.layers, dst.layers):
        dl.dst_nodes.copy_(sl.dst_nodes)
        dl.dst_ts.copy_(sl.dst_ts)
        dl.dst_feat.copy_(sl.dst_feat)

        dl.src_nodes.copy_(sl.src_nodes)
        dl.src_ts.copy_(sl.src_ts)
        dl.src_feat.copy_(sl.src_feat)

        dl.edge_ids.copy_(sl.edge_ids)
        dl.edge_feat.copy_(sl.edge_feat)




def _cpu_stage(cpu_queue: 'queue.Queue[at.AtlasCPUBlock]', *, g, start_idx, end_idx, batch_size,
               neg_sampler, sampler, num_layers, num_neighbors, stop_event: threading.Event):
    curr = start_idx
    while curr < end_idx and not stop_event.is_set():
        last = min(curr + batch_size, end_idx)
        if last <= curr:
            break
        with _nvtx_range("cpu_make_batch"):
            block = at.make_batch(
                g=g,
                start=curr,
                end=last,
                neg_sampler=neg_sampler,
                sampler=sampler,
                num_layers=num_layers,
                num_neighbors=num_neighbors,
            )
        cpu_queue.put(block)
        curr = last
    cpu_queue.put(None)


def _copy_to_device(block, device, *, stream: Optional[torch.cuda.Stream] = None):
    if device.type != 'cuda':
        return block
    ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx:
        with _nvtx_range("h2d_copy"):
            return at.cpu_to_gpu_block(block, device=device)


@contextmanager
def _nvtx_range(name: str):
    if NVTX_ENABLED:
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


@torch.no_grad()
def evaluate(model, g, sampler, neg_sampler, start_idx, end_idx,
             batch_size, num_layers, num_neighbors, device):
    model.eval()
    aps: List[float] = []
    aucs: List[float] = []
    t_eval = tt.start()

    # CPU-only fallback reuses the legacy path
    if device.type != 'cuda':
        curr = start_idx
        while curr < end_idx:
            last = min(curr + batch_size, end_idx)
            if last <= curr:
                break
            cpu_block = at.make_batch(
                g=g,
                start=curr,
                end=last,
                neg_sampler=neg_sampler,
                sampler=sampler,
                num_layers=num_layers,
                num_neighbors=num_neighbors,
            )
            gpu_block = at.cpu_to_gpu_block(cpu_block, device=device)
            pos_scores, neg_scores = model(gpu_block)
            probs = torch.cat([pos_scores, neg_scores], dim=0).sigmoid().cpu()
            labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores),
            ], dim=0).cpu()
            aps.append(average_precision_score(labels.numpy(), probs.numpy()))
            aucs.append(roc_auc_score(labels.numpy(), probs.numpy()))
            curr = last
        ap = 0.0 if not aps else float(np.mean(aps))
        auc = 0.0 if not aucs else float(np.mean(aucs))
        tt.t_eval = tt.elapsed(t_eval)
        print(f"  eval time:{tt.t_eval:.2f}s")
        return ap, auc

    # GPU: reuse the async pipeline from training stage
    cpu_queue_obj: queue.Queue = queue.Queue(maxsize=CPU_QUEUE_SIZE)
    gpu_queue_obj: 'queue.Queue[Optional[at.AtlasGPUBlock]]' = queue.Queue(maxsize=GPU_QUEUE_SIZE)
    cpu_stop = threading.Event()

    cpu_thread = threading.Thread(
        target=_cpu_stage,
        args=(cpu_queue_obj,),
        kwargs=dict(
            g=g,
            start_idx=start_idx,
            end_idx=end_idx,
            batch_size=batch_size,
            neg_sampler=neg_sampler,
            sampler=sampler,
            num_layers=num_layers,
            num_neighbors=num_neighbors,
            stop_event=cpu_stop,
        ),
        daemon=True,
    )
    cpu_thread.start()

    torch.cuda.set_device(device)
    stream_pool = [torch.cuda.Stream(device=device) for _ in range(H2D_STREAMS)]

    def _h2d_stage_eval():
        torch.cuda.set_device(device)
        prefetched = 0
        inflight: List[Tuple[at.AtlasGPUBlock, torch.cuda.Event]] = []
        cpu_done = False
        while True:
            while not cpu_done and (len(inflight) + gpu_queue_obj.qsize()) < GPU_QUEUE_SIZE:
                cpu_block = cpu_queue_obj.get()
                if cpu_block is None:
                    cpu_done = True
                    break
                stream = stream_pool[prefetched % len(stream_pool)]
                prefetched += 1
                gpu_block = _copy_to_device(cpu_block, device, stream=stream)
                event = torch.cuda.Event()
                event.record(stream)
                inflight.append((gpu_block, event))

            ready: List[at.AtlasGPUBlock] = []
            pending: List[Tuple[at.AtlasGPUBlock, torch.cuda.Event]] = []
            for gpu_block, event in inflight:
                if event.query():
                    ready.append(gpu_block)
                else:
                    pending.append((gpu_block, event))
            inflight = pending

            for gpu_block in ready:
                gpu_queue_obj.put(gpu_block)

            if cpu_done and not inflight:
                break

            if not ready:
                time.sleep(0.0005)

        gpu_queue_obj.put(None)

    h2d_thread = threading.Thread(target=_h2d_stage_eval, daemon=True)
    h2d_thread.start()

    while True:
        gpu_block = gpu_queue_obj.get()
        if gpu_block is None:
            break

        with _nvtx_range("gpu_eval"):
            pos_scores, neg_scores = model(gpu_block)
            probs = torch.cat([pos_scores, neg_scores], dim=0).sigmoid().cpu()
            labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores),
            ], dim=0).cpu()
            aps.append(average_precision_score(labels.numpy(), probs.numpy()))
            aucs.append(roc_auc_score(labels.numpy(), probs.numpy()))

    h2d_thread.join()
    cpu_stop.set()
    cpu_thread.join()

    ap = 0.0 if not aps else float(np.mean(aps))
    auc = 0.0 if not aucs else float(np.mean(aucs))
    tt.t_eval = tt.elapsed(t_eval)
    print(f"  eval time:{tt.t_eval:.2f}s")
    return ap, auc


train_end, val_end = support.data_split(g.num_edges(), 0.7, 0.15)
neg_sampler = lambda size: np.random.randint(0, g.num_nodes(), size)

# 
# ----- CUDA Graph preparation -----
USE_CUDAGRAPH = (device.type == 'cuda')
graph = None
static_block: Optional[at.AtlasGPUBlock] = None
static_loss: Optional[torch.Tensor] = None
labels_static: Optional[torch.Tensor] = None
logits_static: Optional[torch.Tensor] = None

if USE_CUDAGRAPH:
    torch.cuda.set_device(device)

    # 1) prepare a dummy batch to get shapes
    dummy_cpu_block = at.make_batch(
        g=g,
        start=0,
        end=BATCH_SIZE,
        neg_sampler=neg_sampler,
        sampler=sampler,
        num_layers=N_LAYERS,
        num_neighbors=N_NBRS,
    )
    dummy_gpu_block = at.cpu_to_gpu_block(dummy_cpu_block, device=device)

    # 2) clone one static_block on GPU; then only do in-place copies
    static_block = clone_static_gpu_block(dummy_gpu_block, device=device)

    # 3) allocate fixed labels/logits buffers on GPU
    with torch.no_grad():
        pos_tmp, neg_tmp = model(dummy_gpu_block)   # only to obtain shapes
        n_pos = pos_tmp.numel()
        n_neg = neg_tmp.numel()
    labels_static = torch.empty(n_pos + n_neg, device=device)
    logits_static = torch.empty(n_pos + n_neg, device=device)
    labels_static[:n_pos].fill_(1.0)
    labels_static[n_pos:].fill_(0.0)

    # 4) create a dedicated capture_stream for warmup and capture
    capture_stream = torch.cuda.Stream(device=device)

    # ---- warmup: run 3 forward + backward passes on the same stream to stabilize allocator ----
    torch.cuda.synchronize(device)           # ensure all previous work is finished
    torch.cuda.set_stream(capture_stream)

    for _ in range(3):
        copy_gpu_block_into_static(dummy_gpu_block, static_block)
        optimizer.zero_grad(set_to_none=True)

        pos_scores, neg_scores = model(static_block)
        logits = torch.cat([pos_scores.view(-1), neg_scores.view(-1)], dim=0)
        warmup_loss = criterion(logits, labels_static)  # directly use labels_static
        warmup_loss.backward()
        optimizer.step()

    capture_stream.synchronize()  # synchronize this stream only

    # ---- 5) capture graph on the same capture_stream ----
    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)

    # explicitly capture using capture_stream here
    with torch.cuda.graph(graph, stream=capture_stream):
        pos_scores, neg_scores = model(static_block)
        logits_static[:n_pos].copy_(pos_scores.view(-1))
        logits_static[n_pos:].copy_(neg_scores.view(-1))

        static_loss = criterion(logits_static, labels_static)
        static_loss.backward()
        optimizer.step()

    # capture finished, restore default stream
    torch.cuda.set_stream(torch.cuda.default_stream(device=device))
    torch.cuda.synchronize(device)




best_epoch = 0
best_ap = 0.0

for epoch in range(EPOCHS):
    nvtx.range_push(f"epoch_{epoch}")
    print(f"epoch {epoch} (async pipeline):")
    torch.cuda.synchronize()
    t_epoch = tt.start()
    model.train()
    epoch_loss = 0.0

    cpu_queue_obj: queue.Queue = queue.Queue(maxsize=CPU_QUEUE_SIZE)
    cpu_stop = threading.Event()

    cpu_thread = threading.Thread(
        target=_cpu_stage,
        args=(cpu_queue_obj,),
        kwargs=dict(
            g=g,
            start_idx=0,
            end_idx=train_end,
            batch_size=BATCH_SIZE,
            neg_sampler=neg_sampler,
            sampler=sampler,
            num_layers=N_LAYERS,
            num_neighbors=N_NBRS,
            stop_event=cpu_stop,
        ),
        daemon=True,
    )
    cpu_thread.start()
    batches = 0

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        stream_pool = [torch.cuda.Stream(device=device) for _ in range(H2D_STREAMS)]
        gpu_queue_obj: 'queue.Queue[Optional[at.AtlasGPUBlock]]' = queue.Queue(maxsize=GPU_QUEUE_SIZE)

        def _h2d_stage():
            torch.cuda.set_device(device)
            prefetched = 0
            inflight: List[Tuple[at.AtlasGPUBlock, torch.cuda.Event]] = []
            cpu_done = False
            while True:
                while not cpu_done and (len(inflight) + gpu_queue_obj.qsize()) < GPU_QUEUE_SIZE:
                    cpu_block = cpu_queue_obj.get()
                    if cpu_block is None:
                        cpu_done = True
                        break
                    stream = stream_pool[prefetched % len(stream_pool)]
                    prefetched += 1
                    gpu_block = _copy_to_device(cpu_block, device, stream=stream)
                    event = torch.cuda.Event()
                    event.record(stream)
                    inflight.append((gpu_block, event))

                ready: List[at.AtlasGPUBlock] = []
                still: List[Tuple[at.AtlasGPUBlock, torch.cuda.Event]] = []
                for gpu_block, event in inflight:
                    if event.query():
                        ready.append(gpu_block)
                    else:
                        still.append((gpu_block, event))
                inflight = still

                for gpu_block in ready:
                    gpu_queue_obj.put(gpu_block)

                if cpu_done and not inflight:
                    break

                if not ready:
                    time.sleep(0.0005)

            gpu_queue_obj.put(None)

        h2d_thread = threading.Thread(target=_h2d_stage, daemon=True)
        h2d_thread.start()

        while True:
            gpu_block = gpu_queue_obj.get()
            if gpu_block is None:
                break

            with _nvtx_range("gpu_compute"):
                if USE_CUDAGRAPH and static_block is not None and gpu_block.batch_size == BATCH_SIZE:
                                        # 1) dynamic → static
                    copy_gpu_block_into_static(gpu_block, static_block)
                    # 2) replay graph
                    graph.replay()  # type: ignore[arg-type]

                    # loss computed inside graph is stored in static_loss
                    loss_val = float(static_loss.detach().cpu()) 
                else:    
                    pos_scores, neg_scores = model(gpu_block)
                    labels = torch.cat([
                        torch.ones_like(pos_scores),
                        torch.zeros_like(neg_scores),
                    ], dim=0)
                    logits = torch.cat([pos_scores, neg_scores], dim=0)

                    loss = criterion(logits, labels)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_val = float(loss.detach().cpu())

            epoch_loss += loss_val
            batches += 1

        h2d_thread.join()
        torch.cuda.synchronize(device)
    else:
        while True:
            cpu_block = cpu_queue_obj.get()
            if cpu_block is None:
                break
            gpu_block = _copy_to_device(cpu_block, device)

            pos_scores, neg_scores = model(gpu_block)
            labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores),
            ], dim=0)
            logits = torch.cat([pos_scores, neg_scores], dim=0)

            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            batches += 1

    cpu_stop.set()
    cpu_thread.join()
    
    tt.t_epoch = tt.elapsed(t_epoch)
    print('  epoch time:{:.2f}s'.format(tt.t_epoch))

    nvtx.range_pop()  # epoch
    nvtx.range_push("validation")
    val_ap, val_auc = evaluate(
        model=model,
        g=g,
        sampler=sampler,
        neg_sampler=neg_sampler,
        start_idx=train_end,
        end_idx=val_end,
        batch_size=BATCH_SIZE,
        num_layers=N_LAYERS,
        num_neighbors=N_NBRS,
        device=device,
    )

    nvtx.range_pop()  # validation

    if epoch == 0 or val_ap > best_ap:
        best_ap = val_ap
        best_epoch = epoch
        torch.save(model.state_dict(), model_path)
        if getattr(g, "mem", None) is not None:
            torch.save(g.mem.backup(), model_mem_path)

    print(f"  batches:{batches} loss:{epoch_loss:.4f} val ap:{val_ap:.4f} val auc:{val_auc:.4f}")

print(f"best model at epoch {best_epoch}")
