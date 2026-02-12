# Distributed Proxy Experiment (R6-style)

This is a lightweight multi-worker/multi-GPU proxy experiment config.

## Run

```bash
cd /home/atlas/tmp_atlas_upload/atlas
python experiments/distributed_proxy/run_train_matrix.py \
  --config experiments/distributed_proxy/train_matrix_distributed_proxy.json \
  --results-dir experiments/distributed_proxy/results

python experiments/distributed_proxy/summarize_runs.py \
  --results-dir experiments/distributed_proxy/results \
  --examples-dir examples
```

Use the summary CSV files for:
- throughput ratio (baseline vs baseline+ATLAS)
- scaling efficiency (if multiple GPUs are available)
- AP/AUC deltas
