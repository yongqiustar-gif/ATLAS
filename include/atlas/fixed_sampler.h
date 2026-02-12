#pragma once

#include "atlas/core.h"
#include "atlas/tcsr.h"
#include "atlas/sampler.h"
#include <vector>

namespace atlas {

class FixedTemporalSampler {
public:
  FixedTemporalSampler(int num_threads, int num_nbrs, bool recent);

  TemporalBlock sample(TCSR &tcsr, py::array_t<IdI32> &nodes, py::array_t<TsF32> &times);

private:
  void sample_layer(TCSR &tcsr, TemporalBlock &block,
                    const IdI32 *nodes_ptr, const TsF32 *times_ptr, size_t size);

  inline void add_neighbor(TCSR &tcsr,
      std::vector<IdI32> *eid, std::vector<TsF32> *ets,
      std::vector<IdI32> *srcnodes, std::vector<IdI32> *dstindex,
      IdI32 &k, IdI32 &dst_idx);

  inline void add_padding(
      std::vector<IdI32> *eid, std::vector<TsF32> *ets,
      std::vector<IdI32> *srcnodes, std::vector<IdI32> *dstindex,
      IdI32 &dst_idx);

  inline void combine_coo(TemporalBlock &block,
      std::vector<IdI32> **eid,
      std::vector<TsF32> **ets,
      std::vector<IdI32> **srcnodes,
      std::vector<IdI32> **dstindex,
      std::vector<IdI32> &out_nodes);

  int _num_threads;
  int _num_nbrs;
  bool _recent;
};

} // namespace atlas
