#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <vector>

#include "atlas/core.h"

namespace atlas {
namespace {

__device__ inline int64_t binary_search_unique(const int64_t* data, int64_t len, int64_t target) {
  int64_t left = 0;
  int64_t right = len - 1;
  while (left <= right) {
    int64_t mid = (left + right) >> 1;
    int64_t value = data[mid];
    if (value == target) {
      return mid;
    }
    if (value < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}

__global__ void map_positions_kernel(const int64_t* __restrict__ flat_index,
                                     int64_t total,
                                     const int64_t* __restrict__ unique,
                                     int64_t unique_len,
                                     int64_t* __restrict__ positions) {
  int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int64_t val = flat_index[idx];
  if (val < 0) {
    positions[idx] = -1;
    return;
  }
  positions[idx] = binary_search_unique(unique, unique_len, val);
}

template <typename scalar_t>
__global__ void gather_from_positions_kernel(const scalar_t* __restrict__ feat_pool,
                                             const int64_t* __restrict__ positions,
                                             scalar_t* __restrict__ out,
                                             int64_t rows,
                                             int64_t feat_dim) {
  int64_t linear = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  int64_t total = rows * feat_dim;
  if (linear >= total) {
    return;
  }
  int64_t row = linear / feat_dim;
  int64_t col = linear % feat_dim;
  int64_t pos = positions[row];
  if (pos < 0) {
    out[linear] = static_cast<scalar_t>(0);
    return;
  }
  out[linear] = feat_pool[pos * feat_dim + col];
}

} // namespace

torch::Tensor recover_feat_from_pool_cuda(torch::Tensor index,
                                          torch::Tensor unique,
                                          torch::Tensor feat_pool) {
  TORCH_CHECK(index.device().is_cuda(), "index must be on CUDA");
  TORCH_CHECK(unique.device().is_cuda(), "unique must be on CUDA");
  TORCH_CHECK(feat_pool.device().is_cuda(), "feat_pool must be on CUDA");
  TORCH_CHECK(index.dtype() == torch::kLong, "index must be int64");
  TORCH_CHECK(unique.dtype() == torch::kLong, "unique must be int64");
  TORCH_CHECK(feat_pool.dim() == 2, "feat_pool must be 2-D");

  auto index_contig = index.contiguous();
  auto unique_contig = unique.contiguous();
  auto feat_contig = feat_pool.contiguous();

  auto feat_dim = feat_contig.size(1);
  auto flat_index = index_contig.view({-1});
  auto rows = flat_index.numel();

  std::vector<int64_t> out_shape(index_contig.sizes().begin(), index_contig.sizes().end());
  out_shape.push_back(feat_dim);
  auto out = torch::zeros(out_shape, feat_contig.options());

  if (rows == 0 || feat_contig.size(0) == 0 || unique_contig.size(0) == 0) {
    return out;
  }

  auto flat_out = out.view({rows, feat_dim});
  auto positions = torch::empty({rows}, index_contig.options());

  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int threads = 256;
  int64_t map_blocks = (rows + threads - 1) / threads;
  if (map_blocks > 0) {
    map_positions_kernel<<<static_cast<int>(map_blocks), threads, 0, stream>>>(
        flat_index.data_ptr<int64_t>(),
        rows,
        unique_contig.data_ptr<int64_t>(),
        unique_contig.numel(),
        positions.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
  }

  int64_t total_elems = rows * feat_dim;
  int64_t gather_blocks = (total_elems + threads - 1) / threads;
  if (gather_blocks > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        feat_contig.scalar_type(),
        "recover_feat_from_pool_cuda",
        [&] {
          gather_from_positions_kernel<scalar_t><<<static_cast<int>(gather_blocks), threads, 0, stream>>>(
              feat_contig.data_ptr<scalar_t>(),
              positions.data_ptr<int64_t>(),
              flat_out.data_ptr<scalar_t>(),
              rows,
              feat_dim);
        });
    AT_CUDA_CHECK(cudaGetLastError());
  }

  return out;
}

} // namespace atlas
