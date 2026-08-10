"""
Substrate integrity audit for every dataset the ablation uses.
===============================================================
Loads the BUILT UKB via CoreEngine (exactly what the ablations consume) and
reports, per dataset, the properties that determine whether results are valid
and comparable -- so we rebuild/rerun once, not per-surprise.

Checks:
  size          : n_docs, n_questions, npart, avg partition size
  titles        : % docs with a non-empty title  (no titles => no relational edges)
  edges         : avg doc-doc degree from structural/title edges vs synthetic kNN
  orphans       : % docs with zero doc-doc edges (unreachable by L3)
  LEAK doc->q   : docs whose neighbours point at question nodes (label backedge)
  LEAK co-gold  : fraction of co-supporting gold PAIRS directly title-linked, vs a
                  random-pair baseline. ~1.0 with near-0 baseline == bridge-edge leak;
                  modestly-elevated == legit content structure.
  pool          : % docs that are gold for >=1 question (==100% => all-gold / no
                  distractors, e.g. decomposition-only musique)
  golds         : avg golds/q, % questions whose golds all resolve to real docs
Writes results/research/audit_substrate.json (+ prints a table).
"""
import os
import json
import logging
import argparse
from collections import defaultdict

import numpy as np

from src.core.engine import CoreEngine

log = logging.getLogger("experiments.audit_substrate")

DATASETS = ["2wiki_clean", "musique_clean", "hotpotqa_clean", "metaqa", "squad"]


def audit(dataset):
    engine = CoreEngine(source=dataset)
    id2idx = engine.node_id_to_idx                      # doc ids only
    docset = set(id2idx)
    docs = engine.nodes
    n = len(docs)
    all_nodes = engine.all_nodes
    q_nodes = [x for x in all_nodes if x.metadata.get("type") == "question"]
    q_ids = set(x.node_id for x in q_nodes)
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    psizes = np.bincount([int(p) for p in engine.partition_map.values()], minlength=npart)

    titled = sum(1 for d in docs if (d.metadata.get("title") or "").strip())
    # doc-doc structural/title edges, synthetic kNN edges, and doc->question backedges
    struct_deg = syn_deg = doc2q = orphans = 0
    doc_adj = defaultdict(set)
    for d in docs:
        s = [x for x in d.neighbors if x in docset]
        q = [x for x in d.neighbors if x in q_ids]
        syn = [x for x in d.metadata.get("synthetic_neighbors", ()) if x in docset]
        struct_deg += len(s); syn_deg += len(syn); doc2q += len(q)
        if not s and not syn:
            orphans += 1
        for x in s:
            doc_adj[d.node_id].add(x)

    # question golds (labels) -> co-gold linkage + pool composition + gold integrity
    gold_of = {}
    gold_docs_all = set()
    resolved_q = dangling_q = 0
    for q in q_nodes:
        ref = [g for g in q.neighbors if g not in q_ids]     # gold refs (exclude q-q)
        golds = [g for g in ref if g in docset]
        gold_of[q.node_id] = golds
        if golds:
            resolved_q += 1
        if len(golds) < len(ref):                            # some gold ids missing from doc set
            dangling_q += 1
        gold_docs_all.update(golds)
    avg_golds = float(np.mean([len(g) for g in gold_of.values() if g])) if gold_of else 0.0

    co_pairs = co_linked = 0
    for golds in gold_of.values():
        for i in range(len(golds)):
            for j in range(i + 1, len(golds)):
                co_pairs += 1
                if golds[j] in doc_adj[golds[i]] or golds[i] in doc_adj[golds[j]]:
                    co_linked += 1
    co_gold_link = (co_linked / co_pairs) if co_pairs else None
    # random-pair baseline: overall probability two random docs are title-linked
    total_struct_edges = sum(len(v) for v in doc_adj.values())   # directed
    rand_base = total_struct_edges / (n * (n - 1)) if n > 1 else 0.0

    out = {
        "dataset": dataset, "n_docs": n, "n_questions": len(q_nodes), "npart": npart,
        "avg_part_size": round(float(psizes.mean()), 1),
        "pct_titled": round(titled / n * 100, 1),
        "deg_struct": round(struct_deg / n, 2), "deg_syn": round(syn_deg / n, 2),
        "pct_orphan_docs": round(orphans / n * 100, 1),
        "leak_doc2q_edges": doc2q,
        "leak_cogold_link_pct": round(co_gold_link * 100, 2) if co_gold_link is not None else None,
        "rand_pair_link_pct": round(rand_base * 100, 4),
        "pct_docs_gold": round(len(gold_docs_all & docset) / n * 100, 1),
        "avg_golds_per_q": round(avg_golds, 2),
        "pct_q_resolved": round(resolved_q / max(len(q_nodes), 1) * 100, 1),
        "n_q_dangling_golds": dangling_q,
    }
    log.info(f"[{dataset}] {json.dumps(out)}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Substrate integrity audit across ablation datasets.")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    a = p.parse_args(argv)
    results = []
    for ds in a.datasets:
        try:
            results.append(audit(ds))
        except Exception as ex:
            log.error(f"[{ds}] FAILED: {type(ex).__name__}: {ex}")
    os.makedirs("results/research", exist_ok=True)
    with open("results/research/audit_substrate.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cols = [("dataset", 15, "s"), ("n_docs", 8, "d"), ("npart", 6, "d"), ("pct_titled", 7, ".1f"),
            ("deg_struct", 11, ".2f"), ("deg_syn", 8, ".2f"), ("pct_orphan_docs", 8, ".1f"),
            ("leak_doc2q_edges", 9, "d"), ("leak_cogold_link_pct", 9, "s"), ("rand_pair_link_pct", 9, "s"),
            ("pct_docs_gold", 8, ".1f"), ("avg_golds_per_q", 8, ".2f")]
    hdr = " ".join(f"{name[:w]:>{w}}" for name, w, _ in cols)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        line = []
        for name, w, fmt in cols:
            v = r.get(name)
            if v is None:
                line.append(f"{'-':>{w}}")
            elif fmt == "s":
                line.append(f"{str(v):>{w}}")
            else:
                line.append(f"{v:>{w}{fmt}}")
        print(" ".join(line))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
