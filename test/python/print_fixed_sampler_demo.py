import numpy as np

from atlas import _c


def build_demo_tcsr():
    """Construct a slightly larger toy graph for manual inspection."""
    edges = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 2],
            [1, 3],
            [2, 3],
            [2, 4],
            [3, 4],
            [4, 5],
            [0, 5],
        ],
        dtype=np.int32,
    )
    times = np.array([0.5, 1.0, 1.2, 1.3, 2.0, 2.4, 2.6, 3.1, 3.3, 3.5], dtype=np.float32)
    num_nodes = 6
    return _c.create_tcsr(edges, times, num_nodes)


def dump_block(block, nodes):
    dst = block.copy_dstindex()
    src = block.copy_srcnodes()
    eid = block.copy_eid()
    ets = block.copy_ets()

    print("=== Raw arrays ===")
    print("dstindex:", dst)
    print("srcnodes:", src)
    print("eid:     ", eid)
    print("etime:   ", ets)

    print("\n=== Grouped by queried node ===")
    for i, nid in enumerate(nodes):
        mask = dst == i
        print(f"Node {int(nid)} @ slot {i}:")
        for src_id, edge_id, edge_ts in zip(src[mask], eid[mask], ets[mask]):
            print(f"  src={int(src_id):2d} eid={int(edge_id):2d} ts={edge_ts:.2f}")
        if mask.sum() == 0:
            print("  (no rows)")


def main():
    tcsr = build_demo_tcsr()
    sampler = _c.FixedTemporalSampler(4, 3, True)

    # Sample five destinations; node 5 has only one historical neighbor
    nodes = np.array([0, 1, 2, 4, 5], dtype=np.int32)
    query_times = np.array([4.0, 3.0, 3.5, 4.0, 4.0], dtype=np.float32)

    block = sampler.sample(tcsr, nodes, query_times)
    dump_block(block, nodes)


if __name__ == "__main__":
    main()
