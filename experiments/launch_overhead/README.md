# Launch Overhead Experiment

## Build

```bash
cd /home/atlas/tmp_atlas_upload/atlas
nvcc -O3 -std=c++17 -arch=compute_89 -code=sm_89 \
  experiments/launch_overhead/cuda_launch_overhead_multimethod.cu \
  -o experiments/launch_overhead/cuda_launch_overhead_multimethod
```

## Run

```bash
experiments/launch_overhead/cuda_launch_overhead_multimethod \
  200000 20000 1000 50000
```

Outputs include:
- `methodA_*`: async enqueue launch overhead
- `methodB_*`: launch+sync step overhead
- `methodC_*`: tiny-kernel event envelope
