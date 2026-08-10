"""
Re-partition an existing UKB source at a chosen granularity (in place).
=======================================================================
Applies the agreed partition design (target docs/partition, METIS on
structural+kNN edges) to a source's existing UKB — reusing its embeddings, so
it's fast (no re-encode). Overwrites partition_map.json, centroids.index,
graph.pt. Used to roll the partition-ablation decision out to every dataset
uniformly (2wiki_clean, musique_clean, metaqa, squad).
"""
import os
import argparse
import logging

from src.core.engine import CoreEngine
from src.core import indexers
from src.experiments.overlap_retrain import _reconstruct

log = logging.getLogger("pipeline.repartition")


def repartition(source, target):
    engine = CoreEngine(source=source)
    nv = _reconstruct(engine.node_index)              # embeddings, aligned to engine.nodes order
    out_dir = os.path.join("data", "ukb_storage", source)
    nodes = engine.nodes                              # doc StandardNodes (questions excluded)
    log.info(f"Re-partitioning '{source}' ({len(nodes)} docs) at ~{target}/partition (structural+kNN)…")
    G_nx = indexers.build_pyg_graph(nodes, nv, out_dir)                 # structural + kNN -> graph.pt
    parts = indexers.build_partition_map(nodes, G_nx, out_dir, target)  # METIS
    indexers.build_faiss_centroid_index(nodes, parts, nv, out_dir)      # rebuilt centroids
    npart = int(max(parts)) + 1 if parts else 0
    log.info(f"  '{source}' -> {npart} partitions")
    return npart


def main(argv=None):
    p = argparse.ArgumentParser(description="Re-partition existing UKB sources at a chosen granularity.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean", "metaqa", "squad"])
    p.add_argument("--target_per_partition", type=int, default=100)
    a = p.parse_args(argv)
    counts = {}
    for ds in a.datasets:
        counts[ds] = repartition(ds, a.target_per_partition)
    log.info(f"DONE repartition @ {a.target_per_partition}/part: {counts}")
    print("PARTITION COUNTS:", counts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
