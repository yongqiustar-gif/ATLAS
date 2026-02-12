import numpy as np

from atlas import _c


def build_simple_tcsr():
	"""Small helper graph with four nodes and three timestamped edges."""
	edges = np.array(
		[
			[0, 1],
			[1, 2],
			[0, 2],
		],
		dtype=np.int32,
	)
	times = np.array([1.0, 2.0, 3.0], dtype=np.float32)
	num_nodes = 4  # node 3 is intentionally isolated
	return _c.create_tcsr(edges, times, num_nodes)


def test_fixed_sampler_returns_padded_neighbors():
	tcsr = build_simple_tcsr()
	sampler = _c.FixedTemporalSampler(2, 2, True)

	# Query node 0 (has two valid neighbors) and node 3 (isolated)
	nodes = np.array([0, 3], dtype=np.int32)
	query_times = np.array([4.0, 4.0], dtype=np.float32)

	block = sampler.sample(tcsr, nodes, query_times)

	dst_index = block.copy_dstindex()
	src_nodes = block.copy_srcnodes()
	edge_ids = block.copy_eid()
	edge_ts = block.copy_ets()

	# Each node should contribute exactly num_nbrs rows
	assert len(dst_index) == nodes.size * 2
	assert len(src_nodes) == len(dst_index)

	# First node (dst_index == 0) has real neighbors
	mask_node0 = dst_index == 0
	assert mask_node0.sum() == 2
	assert np.all(src_nodes[mask_node0] >= 0)
	assert np.all(edge_ids[mask_node0] >= 0)

	# Second node (dst_index == 1) should be padded with dummy entries
	mask_node1 = dst_index == 1
	assert mask_node1.sum() == 2
	assert np.all(src_nodes[mask_node1] == -1)
	assert np.all(edge_ids[mask_node1] == -1)
	assert np.allclose(edge_ts[mask_node1], 0.0)
