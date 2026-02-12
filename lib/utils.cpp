#include "atlas/utils.h"
#include <vector>
#include <immintrin.h> // SIMD intrinsics

namespace atlas {




  torch::Tensor pinned_gather(const torch::Tensor& feat, const torch::Tensor& idx) {
      TORCH_CHECK(feat.device().is_cpu(), "feat must be on CPU");
      TORCH_CHECK(idx.device().is_cpu(), "idx must be on CPU");
      TORCH_CHECK(idx.dtype() == torch::kLong, "idx must be int64");
      TORCH_CHECK(feat.dtype() == torch::kFloat32, "feat must be float32");

      auto M = idx.numel();
      auto D = feat.size(1);
      auto out = torch::empty({M, D}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));

      auto feat_a = feat.accessor<float, 2>();
      auto idx_a = idx.accessor<int64_t, 1>();
      auto out_a = out.accessor<float, 2>();

      #pragma omp parallel for
      for (int64_t i = 0; i < M; ++i) {
          int64_t src = idx_a[i];
          int64_t d = 0;
          // AVX512 path: process 16 floats per iteration
          for (; d + 16 <= D; d += 16) {
              __m512 v = _mm512_loadu_ps(&feat_a[src][d]);
              _mm512_storeu_ps(&out_a[i][d], v);
          }
          // Remaining elements
          for (; d < D; ++d) {
              out_a[i][d] = feat_a[src][d];
          }
      }
      return out;
  }


py::array_t<IdI32> find_latest_uniq(py::array_t<IdI32> &uniq, py::array_t<IdI32> &nodes, py::array_t<TsF32> &times) {
  ssize_t num_uniq = uniq.size();
  ssize_t num_nodes = nodes.size();

  auto *index = new std::vector<IdI32>;
  index->resize(num_uniq);

  auto *uniq_ptr = static_cast<IdI32 *>(uniq.request().ptr);
  auto *node_ptr = static_cast<IdI32 *>(nodes.request().ptr);
  auto *time_ptr = static_cast<TsF32 *>(times.request().ptr);

  #pragma omp parallel for schedule(static)
  for (ssize_t i = 0; i < num_uniq; i++) {
    IdI32 nid = uniq_ptr[i];
    TsF32 max = -1.0f;
    for (ssize_t j = num_nodes - 1; j >= 0; j--) {
      if (node_ptr[j] == nid && time_ptr[j] > max) {
        max = time_ptr[j];
        (*index)[i] = j;
      }
    }
  }

  py::array_t<IdI32> res = to_pyarray_owned(index);
  return res;
}

} // namespace atlas
