# ATLAS

## Install

```bash
conda create -n atlas python=3.12
conda activate atlas

# Install a CUDA-compatible PyTorch build (>2.0)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

cd atlas
pip install -e . --no-build-isolation
pip install scikit-learn pandas
```

## Data

```bash
cd examples
chmod +x download-data.sh
./download-data.sh
```

## Retained Experiment Set

This repository intentionally keeps only two focused experiment groups under `experiments/`:

1. `experiments/launch_overhead/`
   - CUDA kernel launch overhead microbenchmark
   - CUDA Graph overhead accounting helpers

2. `experiments/distributed_proxy/`
   - R6-style distributed composability proxy runs
   - run/summarize scripts + config

See each subdirectory README for commands.
