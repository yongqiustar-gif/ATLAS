#include "atlas/core.h"
// #include "atlas/cache.h"
// #include "atlas/dedup.h"
#include "atlas/sampler.h"
#include "atlas/fixed_sampler.h"
#include "atlas/tcsr.h"
#include "atlas/utils.h"

using namespace atlas;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<TCSR>(m, "TCSR")
    .def(py::init<>())
    .def_property_readonly("ind", &TCSR::ind_view)
    .def_property_readonly("nbr", &TCSR::nbr_view)
    .def_property_readonly("eid", &TCSR::eid_view)
    .def_property_readonly("ets", &TCSR::ets_view);

  py::class_<TemporalBlock>(m, "TemporalBlock")
    .def(py::init<>())
    .def("copy_eid", &TemporalBlock::eid_copy)
    .def("copy_ets", &TemporalBlock::ets_copy)
    .def("copy_srcnodes", &TemporalBlock::srcnodes_copy)
    .def("copy_dstindex", &TemporalBlock::dstindex_copy)
    .def("num_edges", [](const TemporalBlock &b) { return b.num_edges; });

  py::class_<TemporalSampler>(m, "TemporalSampler")
    .def(py::init<int, int, bool>())
    .def("sample", &TemporalSampler::sample);

  py::class_<FixedTemporalSampler>(m, "FixedTemporalSampler")
    .def(py::init<int, int, bool>())
    .def("sample", &FixedTemporalSampler::sample);

//   py::class_<EmbedTable>(m, "EmbedTable")
//     .def(py::init<ssize_t, ssize_t>())
//     // .def("_table", [](const EmbedTable &et) { return et._table; })
//     // .def("_keys", [](const EmbedTable &et) { return et._keys; })
//     // .def("_map", [](const EmbedTable &et) { return et._key2idx; })
//     .def("lookup", &EmbedTable::lookup, "atlas::EmbedTable::lookup")
//     .def("store", &EmbedTable::store, "atlas::EmbedTable::store");

  m.def("create_tcsr", &create_tcsr, "atlas::create_tcsr");
  m.def("pinned_gather", &atlas::pinned_gather, "atlas::pinned_gather");
  m.def("recover_feat_from_pool", &atlas::recover_feat_from_pool_cuda, "atlas::recover_feat_from_pool");

//   m.def("dedup_targets", &dedup_targets, "atlas::dedup_targets");
//   m.def("find_latest_uniq", &find_latest_uniq, "atlas::find_latest_uniq");
//   // m.def("find_last_message", &find_last_message, "atlas::find_last_message");
//   m.def("find_dedup_time_hits", &find_dedup_time_hits, "atlas::find_dedup_time_hits");
//   m.def("compute_cache_keys", &compute_cache_keys, "atlas::compute_cache_keys");

  // m.def("dedup_indices", &dedup_indices, "atlas::dedup_indices");
  // m.def("index_pinned", &index_pinned, "atlas::index_pinned");
}
