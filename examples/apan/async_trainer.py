import os
import queue
import threading
import time
from sklearn.metrics import average_precision_score, roc_auc_score
from contextlib import contextmanager, nullcontext



import numpy as np
import torch

from typing import Callable, List, Optional, Tuple

import atlas as at
from atlas import tt

from torch.cuda import nvtx

NVTX_ENABLED = torch.cuda.is_available()


def _nvtx_push(name: str) -> None:
    if NVTX_ENABLED:
        nvtx.range_push(name)


def _nvtx_pop() -> None:
    if NVTX_ENABLED:
        nvtx.range_pop()


@contextmanager
def _nvtx_range(name: str):
    _nvtx_push(name)
    try:
        yield
    finally:
        _nvtx_pop()

def configure_cpu_threads(requested: int) -> int:
    """Clamp and propagate thread counts to the common BLAS env vars."""
    logical = os.cpu_count() or 16
    target = logical if requested <= 0 else min(requested, logical)
    target = max(1, target)
    env_val = str(target)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, env_val)
    torch.set_num_threads(target)
    torch.set_num_interop_threads(max(1, min(4, target // 4 or 1)))
    return target

class AsyncPipelineTrainer:

    def __init__(
            self,
            g: at.TGraph,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            sampler: at.FixedTSampler,
            neg_sampler: Callable[[int], np.ndarray],
            *,
            epochs: int,
            batch_size: int,
            train_end: int,
            val_end: int,
            num_layers: int,
            num_neighbors: int,
            device: torch.device,
            model_path: str,
            model_mem_path: Optional[str],
            cpu_queue_size: int = 8,
            gpu_queue_size: int = 4,
            h2d_streams: int = 1,
            warmup_steps: int = 3,
            use_cuda_graph: bool = True,
            ) -> None:
        self.g = g
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.sampler = sampler
        self.neg_sampler = neg_sampler
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_end = train_end
        self.val_end = val_end
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.device = device
        self.model_path = model_path
        self.model_mem_path = model_mem_path
        self.cpu_queue_size = max(1, cpu_queue_size)
        self.gpu_queue_size = max(1, gpu_queue_size)
        self.h2d_streams = max(1, h2d_streams)
        self.warmup_steps = warmup_steps
        self.cuda_graph_enabled = use_cuda_graph and device.type == 'cuda'

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_block: Optional[at.AtlasGPUBlock] = None
        self.static_loss: Optional[torch.Tensor] = None
        self.lables_static: Optional[torch.Tensor] = None
        self.logits_static: Optional[torch.Tensor] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
        self.pos_count: int = 0
        self.neg_count: int = 0
    
    def train(self) -> None:
        if self.cuda_graph_enabled:
            self._prepare_cuda_graph()
        
        nvtx.range_push("running training")
        best_epoch = 0
        best_ap = 0.0
        for epoch in range(self.epochs):
            print(f"epoch {epoch} (async pipeline):")
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            
            t_epoch = tt.start()
            self.model.train()

            self.model.memory.zero_()
            self.model.memory_ts.zero_()



            epoch_loss = 0.0
            batches = 0

            cpu_queue_obj: queue.Queue = queue.Queue(maxsize=self.cpu_queue_size)
            cpu_stop = threading.Event()
            cpu_thread = threading.Thread(
                target=self._cpu_stage,
                args=(cpu_queue_obj,),
                kwargs=dict(
                    g=self.g,
                    start_idx=0,
                    end_idx=self.train_end,
                    batch_size=self.batch_size,
                    neg_sampler=self.neg_sampler,
                    sampler=self.sampler,
                    num_layers=self.num_layers,
                    num_neighbors=self.num_neighbors,
                    stop_event=cpu_stop,
                ),
                daemon=True,
            )
            cpu_thread.start()

            if self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
                stream_pool = [torch.cuda.Stream(device=self.device) for _ in range(self.h2d_streams)]
                gpu_queue_obj: "queue.Queue[Optional[at.AtlasGPUBlock]]" = queue.Queue(maxsize=self.gpu_queue_size)

                def _h2d_stage():
                    torch.cuda.set_device(self.device)
                    prefetched = 0
                    inflight: List[Tuple[at.AtlasGPUBlock, torch.cuda.Event]] = []
                    cpu_done = False
                    while True:
                        while not cpu_done and (len(inflight) + gpu_queue_obj.qsize()) < self.gpu_queue_size:
                            cpu_block = cpu_queue_obj.get()
                            if cpu_block is None:
                                cpu_done = True
                                break
                            stream = stream_pool[prefetched % len(stream_pool)]
                            prefetched += 1
                            nvtx.range_push("h2d_copy")
                            gpu_block = self._copy_to_device(cpu_block, stream=stream)
                            nvtx.range_pop()
                            event = torch.cuda.Event()
                            event.record(stream)
                            inflight.append((gpu_block, event))

                        ready: List[at.AtlasGPUBlock] = []
                        pending: List[Tuple[at.AtlasGPUBlock, torch.cuda.Event]] = []
                        for block, event in inflight:
                            if event.query():
                                ready.append(block)
                            else:
                                pending.append((block, event))
                        inflight = pending

                        for block in ready:
                            gpu_queue_obj.put(block)

                        if cpu_done and not inflight:
                            break

                        if not ready:
                            time.sleep(0.0005)

                    gpu_queue_obj.put(None)

                h2d_thread = threading.Thread(target=_h2d_stage, daemon=True)
                h2d_thread.start()

                # gpu compute
                while True:
                    gpu_block = gpu_queue_obj.get()
                    if gpu_block is None:
                        break
                    nvtx.range_push("gpu_compute")
                    loss_val = self._train_on_block(gpu_block)
                    nvtx.range_pop()
                    epoch_loss += loss_val
                    batches += 1
                h2d_thread.join()
                torch.cuda.synchronize(self.device)
            else:
                while True:
                    cpu_block = cpu_queue_obj.get()
                    if cpu_block is None:
                        break
                    gpu_block = self._copy_to_device(cpu_block)
                    loss_val = self._train_on_block(gpu_block)
                    epoch_loss += loss_val
                    batches += 1

            cpu_stop.set()
            cpu_thread.join()

            tt.t_epoch = tt.elapsed(t_epoch)
            print(f"  train epoch time: {tt.t_epoch:.2f}s")

            with _nvtx_range("validation"):
                val_ap, val_auc = self.evaluate(self.train_end, self.val_end)

            if epoch == 0 or val_ap > best_ap:
                best_ap = val_ap
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_path)
                if self.g._mem is not None and self.model_mem_path is not None:
                    torch.save(self.g.mem.backup(), self.model_mem_path)

            print(f"  batches:{batches} loss:{epoch_loss:.4f} val ap:{val_ap:.4f} val auc:{val_auc:.4f}")
        nvtx.range_pop()  # training_loop
        print(f"best model at epoch {best_epoch}")

    def _prepare_cuda_graph(self) -> None:
        torch.cuda.set_device(self.device)
        dummy_cpu_block = at.make_batch(
            g=self.g,
            start=0,
            end=min(self.batch_size, self.g.num_edges()),
            neg_sampler=self.neg_sampler,
            sampler=self.sampler,
            num_layers=self.num_layers,
            num_neighbors=self.num_neighbors,
        )
        dummy_gpu_block = at.cpu_to_gpu_block(dummy_cpu_block, device=self.device)
        self.static_block = self._clone_static_gpu_block(dummy_gpu_block)

        with torch.no_grad():
            pos_tmp, neg_tmp = self.model(dummy_gpu_block)
            self.pos_count = pos_tmp.numel()
            self.neg_count = neg_tmp.numel()

        self.labels_static = torch.empty(self.pos_count + self.neg_count, device=self.device)
        self.logits_static = torch.empty(self.pos_count + self.neg_count, device=self.device)
        self.labels_static[: self.pos_count].fill_(1.0)
        self.labels_static[self.pos_count :].fill_(0.0)

        self.capture_stream = torch.cuda.Stream(device=self.device)
        torch.cuda.synchronize(self.device)
        torch.cuda.set_stream(self.capture_stream)

        for _ in range(self.warmup_steps):
            self._copy_gpu_block_into_static(dummy_gpu_block, self.static_block, warmup=True)
            self.optimizer.zero_grad(set_to_none=True)
            pos_scores, neg_scores = self.model(self.static_block)
            logits = torch.cat([pos_scores.view(-1), neg_scores.view(-1)], dim=0)
            warmup_loss = self.criterion(logits, self.labels_static)
            warmup_loss.backward()
            self.optimizer.step()

        self.capture_stream.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            pos_scores, neg_scores = self.model(self.static_block)
            self.logits_static[: self.pos_count].copy_(pos_scores.view(-1))
            self.logits_static[self.pos_count :].copy_(neg_scores.view(-1))
            self.static_loss = self.criterion(self.logits_static, self.labels_static)
            self.static_loss.backward()
            self.optimizer.step()

        torch.cuda.set_stream(torch.cuda.default_stream(device=self.device))
        torch.cuda.synchronize(self.device)

    @staticmethod
    def _cpu_stage(
            cpu_queue_obj: 'queue.Queue[Optional[at.AtlasGPUBlock]]',
            *,
            g: at.TGraph,
            start_idx: int,
            end_idx: int,
            batch_size: int,
            neg_sampler: Callable[[int], np.ndarray],
            sampler: at.FixedTSampler,
            num_layers: int,
            num_neighbors: int,
            stop_event: threading.Event
    ) -> None:
        curr = start_idx
        while curr < end_idx and not stop_event.is_set():
            last = min(curr + batch_size, end_idx)
            if last <= curr:
                break
            nvtx.range_push("cpu_stage: make batch")
            block = at.make_batch_index(
                g=g,
                start=curr,
                end=last,
                neg_sampler=neg_sampler,
                sampler=sampler,
                num_layers=num_layers,
                num_neighbors=num_neighbors,
            )
            nvtx.range_pop()  # cpu_stage: make batch
            cpu_queue_obj.put(block)
            curr = last
        cpu_queue_obj.put(None)  # signal end of data

    def _copy_to_device(
        self,
        block: at.AtlasCPUBlock,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> at.AtlasGPUBlock:
        if self.device.type != "cuda":
            return at.cpu_to_gpu_block(block, device=self.device)
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            return at.cpu_to_gpu_block(block, device=self.device)

    def _train_on_block(self, gpu_block: at.AtlasGPUBlock) -> float:
        if (
            self.cuda_graph_enabled
            and self.graph is not None
            and self.static_block is not None
            and self.static_loss is not None
            and gpu_block.batch_size == self.batch_size
        ):
            self._copy_gpu_block_into_static(gpu_block, self.static_block)
            self.graph.replay()
            return float(self.static_loss.detach().cpu())
        loss = self._run_step(gpu_block)
        return float(loss.detach().cpu())

    def _run_step(self, gpu_block: at.AtlasGPUBlock) -> torch.Tensor:
        gpu_block = at.recover_block(gpu_block)
        pos_scores, neg_scores = self.model(gpu_block)
        labels = torch.cat(
            [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)],
            dim=0,
        )
        logits = torch.cat([pos_scores, neg_scores], dim=0)
        loss = self.criterion(logits, labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss

    def test(self) -> None:
        print("loading saved checkpoint and testing model...")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        if self.g._mem is not None and self.model_mem_path is not None and os.path.exists(self.model_mem_path):
            self.g._mem.restore(torch.load(self.model_mem_path, map_location=self.device))
        ap, auc = self.evaluate(self.val_end, self.g.num_edges())
        print(f"  test: AP:{ap:.4f} AUC:{auc:.4f}")

    @torch.no_grad()
    def evaluate(self, start_idx: int, end_idx: Optional[int]) -> Tuple[float, float]:
        self.model.eval()
        aps: List[float] = []
        aucs: List[float] = []
        t_eval = tt.start()

        if self.device.type != "cuda":
            curr = start_idx
            while curr < (end_idx or self.g.num_edges()):
                last = min(curr + self.batch_size, end_idx or self.g.num_edges())
                if last <= curr:
                    break
                cpu_block = at.make_batch(
                    g=self.g,
                    start=curr,
                    end=last,
                    neg_sampler=self.neg_sampler,
                    sampler=self.sampler,
                    num_layers=self.num_layers,
                    num_neighbors=self.num_neighbors,
                )
                gpu_block = at.cpu_to_gpu_block(cpu_block, device=self.device)
                pos_scores, neg_scores = self.model(gpu_block)
                probs = torch.cat([pos_scores, neg_scores], dim=0).sigmoid().cpu()
                labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0).cpu()
                aps.append(average_precision_score(labels.numpy(), probs.numpy()))
                aucs.append(roc_auc_score(labels.numpy(), probs.numpy()))
                curr = last
        else:
            cpu_queue_obj: queue.Queue = queue.Queue(maxsize=self.cpu_queue_size)
            gpu_queue_obj: "queue.Queue[Optional[at.AtlasGPUBlock]]" = queue.Queue(maxsize=self.gpu_queue_size)
            cpu_stop = threading.Event()

            cpu_thread = threading.Thread(
                target=self._cpu_stage,
                args=(cpu_queue_obj,),
                kwargs=dict(
                    g=self.g,
                    start_idx=start_idx,
                    end_idx=end_idx or self.g.num_edges(),
                    batch_size=self.batch_size,
                    neg_sampler=self.neg_sampler,
                    sampler=self.sampler,
                    num_layers=self.num_layers,
                    num_neighbors=self.num_neighbors,
                    stop_event=cpu_stop,
                ),
                daemon=True,
            )
            cpu_thread.start()

            torch.cuda.set_device(self.device)
            stream_pool = [torch.cuda.Stream(device=self.device) for _ in range(self.h2d_streams)]

            def _h2d_stage_eval():
                torch.cuda.set_device(self.device)
                prefetched = 0
                inflight: List[Tuple[at.AtlasGPUBlock, torch.cuda.Event]] = []
                cpu_done = False
                while True:
                    while not cpu_done and (len(inflight) + gpu_queue_obj.qsize()) < self.gpu_queue_size:
                        cpu_block = cpu_queue_obj.get()
                        if cpu_block is None:
                            cpu_done = True
                            break
                        stream = stream_pool[prefetched % len(stream_pool)]
                        prefetched += 1
                        gpu_block = self._copy_to_device(cpu_block, stream=stream)
                        event = torch.cuda.Event()
                        event.record(stream)
                        inflight.append((gpu_block, event))

                    ready: List[at.AtlasGPUBlock] = []
                    pending: List[Tuple[at.AtlasGPUBlock, torch.cuda.Event]] = []
                    for block, event in inflight:
                        if event.query():
                            ready.append(block)
                        else:
                            pending.append((block, event))
                    inflight = pending

                    for block in ready:
                        gpu_queue_obj.put(block)

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
                    gpu_block = at.recover_block(gpu_block)
                    pos_scores, neg_scores = self.model(gpu_block)
                    probs = torch.cat([pos_scores, neg_scores], dim=0).sigmoid().cpu()
                    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0).cpu()
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



    def _clone_static_gpu_block(self, template: at.AtlasGPUBlock) -> at.AtlasGPUBlock:
        def empty_like(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x, device=self.device)

        static_layers: List[at.AtlasGPULayer] = []
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

    @staticmethod
    def _copy_gpu_block_into_static(src: at.AtlasGPUBlock, dst: at.AtlasGPUBlock, warmup: bool = False) -> None:
        if not warmup:
            src = at.recover_block(src)
        dst.root_src.copy_(src.root_src)
        dst.root_dst.copy_(src.root_dst)
        dst.root_neg.copy_(src.root_neg)
        dst.root_ts.copy_(src.root_ts)
        dst.root_src_feat.copy_(src.root_src_feat)
        dst.root_dst_feat.copy_(src.root_dst_feat)
        dst.root_neg_feat.copy_(src.root_neg_feat)
        dst.root_edge_feat.copy_(src.root_edge_feat)

        for sl, dl in zip(src.layers, dst.layers):
            dl.dst_nodes.copy_(sl.dst_nodes)
            dl.dst_ts.copy_(sl.dst_ts)
            dl.dst_feat.copy_(sl.dst_feat)
            dl.src_nodes.copy_(sl.src_nodes)
            dl.src_ts.copy_(sl.src_ts)
            dl.src_feat.copy_(sl.src_feat)
            dl.edge_ids.copy_(sl.edge_ids)
            dl.edge_feat.copy_(sl.edge_feat)

        dst.root_src.copy_(src.root_src)
        dst.root_dst.copy_(src.root_dst)
        dst.root_neg.copy_(src.root_neg)
        dst.root_ts.copy_(src.root_ts)
        dst.root_src_feat.copy_(src.root_src_feat)
        dst.root_dst_feat.copy_(src.root_dst_feat)
        dst.root_neg_feat.copy_(src.root_neg_feat)
        dst.root_edge_feat.copy_(src.root_edge_feat)

        for l in range(src.num_layers):
            src_lyr = src.layers[l]
            dst_lyr = dst.layers[l]
            dst_lyr.dst_nodes.copy_(src_lyr.dst_nodes)
            dst_lyr.dst_ts.copy_(src_lyr.dst_ts)
            dst_lyr.dst_feat.copy_(src_lyr.dst_feat)
            dst_lyr.src_nodes.copy_(src_lyr.src_nodes)
            dst_lyr.src_ts.copy_(src_lyr.src_ts)
            dst_lyr.src_feat.copy_(src_lyr.src_feat)
            dst_lyr.edge_ids.copy_(src_lyr.edge_ids)
            dst_lyr.edge_feat.copy_(src_lyr.edge_feat)
