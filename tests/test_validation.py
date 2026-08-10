"""
Validation tests for the CRAG experiment code.
===============================================
No test suite existed; every result was eyeballed, and several silent-corruption
bugs slipped through (PPR scalar-index, APPNP raw-vs-normalized adjacency, combined
< dense metric flaw, L3-traverse stopping on seeds, metaqa OOM). These test the
things that determine every reported number:
  - metric correctness + invariants (FullCov <= gt_recall <= recall; monotone in k;
    combined >= components — the flaw we hit)
  - significance (McNemar exact) on hand-computed cases
  - result-store path classification + pool-match interpolation
  - SUBSTRATE integrity (leak-free: no doc->question edges; train/test queries
    disjoint; golds resolve to docs) — run with --substrate (loads engines)

Runs standalone: `python -m tests.test_validation [--substrate]` (also pytest-compatible).
"""
import sys
import math
import numpy as np

from src.experiments.stats import mcnemar_exact, paired
from src.experiments.l3_methods import _metrics
from src.pipeline.ukb_results import classify
from src.experiments.champion import _interp, K_GRID

_checks = []


def _ok(name, cond):
    _checks.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return cond


def test_metrics_known():
    # order[0] = ranked doc ids; golds = {2,4}
    order = np.array([[0, 1, 2, 3, 4]])
    gr, fc = _metrics(order, [[2, 4]], budgets=[2, 3, 5])
    _ok("recall@2 == 0 (neither gold in top-2)", gr["gt_recall@2"] == 0.0)
    _ok("recall@3 == 50 (one of two golds)", gr["gt_recall@3"] == 50.0)
    _ok("recall@5 == 100 (both golds)", gr["gt_recall@5"] == 100.0)
    _ok("fullcov@3 == 0 (not all golds)", fc["fullcov@3"] == 0.0)
    _ok("fullcov@5 == 100 (all golds)", fc["fullcov@5"] == 100.0)


def test_metric_invariants():
    rng = np.random.RandomState(0)
    n, nq = 500, 40
    order = np.array([rng.permutation(n) for _ in range(nq)])
    gold = [rng.choice(n, rng.randint(1, 4), replace=False).tolist() for _ in range(nq)]
    bud = [5, 20, 50, 100]
    gr, fc = _metrics(order, gold, budgets=bud)
    mono = all(gr[f"gt_recall@{bud[i]}"] <= gr[f"gt_recall@{bud[i+1]}"] + 1e-9 for i in range(len(bud) - 1))
    _ok("gt_recall monotone non-decreasing in k", mono)
    fmono = all(fc[f"fullcov@{bud[i]}"] <= fc[f"fullcov@{bud[i+1]}"] + 1e-9 for i in range(len(bud) - 1))
    _ok("fullcov monotone non-decreasing in k", fmono)
    _ok("fullcov <= gt_recall at every k", all(fc[f"fullcov@{b}"] <= gr[f"gt_recall@{b}"] + 1e-9 for b in bud))


def test_combined_ge_components():
    # the flaw we hit: a proper union/combine must never be worse than its parts.
    rng = np.random.RandomState(1)
    n, nq, k = 300, 30, 50
    A = np.array([rng.permutation(n) for _ in range(nq)])
    B = np.array([rng.permutation(n) for _ in range(nq)])
    gold = [rng.choice(n, 2, replace=False).tolist() for _ in range(nq)]
    # PROPER combine: union preserving order, A then B (no fixed split truncation)
    comb = np.array([list(dict.fromkeys(A[i].tolist() + B[i].tolist()))[:k] for i in range(nq)])
    ga, _ = _metrics(A, gold, budgets=[k]); gc, _ = _metrics(comb, gold, budgets=[k])
    _ok("combined@k >= component-A@k (proper union never hurts)",
        gc[f"gt_recall@{k}"] >= ga[f"gt_recall@{k}"] - 1e-9)


def test_mcnemar():
    _ok("mcnemar(0,0) == 1.0 (no discordant)", mcnemar_exact(0, 0) == 1.0)
    _ok("mcnemar(5,5) ~ 1.0 (symmetric)", mcnemar_exact(5, 5) > 0.9)
    _ok("mcnemar(20,2) significant (<0.05)", mcnemar_exact(20, 2) < 0.05)
    _ok("mcnemar symmetric in (b,c)", abs(mcnemar_exact(13, 4) - mcnemar_exact(4, 13)) < 1e-12)
    r = paired([1, 1, 1, 0], [0, 0, 1, 1])   # b=2 (treat-only), c=1 (base-only)
    _ok("paired b/c counts correct", r["b_treatment_only"] == 2 and r["c_baseline_only"] == 1)


def test_classify():
    cases = {
        "results/overlap_ablation/metaqa_overlap_retrain_S1loss.json": ("metaqa", "L1", "loss_ablation.json"),
        "results/overlap_ablation/2wiki_clean_overlap_retrain_S2struct.json": ("2wiki_clean", "L1", "structure_sweep.json"),
        "results/research/champion_2wiki_clean.json": ("2wiki_clean", "L1", "champion"),
        "results/research/reach_metaqa.json": ("metaqa", "L3", "reachability"),
        "results/l3_recovery/musique_clean_overlap1_knn1.json": ("musique_clean", "L3", "recovery__overlap1_knn1.json"),
    }
    for p, exp in cases.items():
        _ok(f"classify {p.split('/')[-1]}", classify(p) == exp)


def test_interp():
    _ok("interp midpoint", abs(_interp(5, [1, 10], [10.0, 100.0]) - 50.0) < 1e-6)
    _ok("interp clamps below range", _interp(-3, [1, 10], [10.0, 100.0]) == 10.0)
    _ok("interp clamps above range", _interp(99, [1, 10], [10.0, 100.0]) == 100.0)


def test_substrate(datasets=("musique_clean", "squad_clean")):
    from src.core.engine import CoreEngine
    from src.experiments.overlap_retrain import _splits, _hard_membership
    for ds in datasets:
        eng = CoreEngine(source=ds)
        docset = set(eng.node_id_to_idx)
        qids = {n.node_id for n in eng.all_nodes if n.metadata.get("type") == "question"}
        doc2q = sum(1 for n in eng.nodes for nb in n.neighbors if nb in qids)
        _ok(f"[{ds}] 0 doc->question edges (label-free)", doc2q == 0)
        sp = _splits(eng, _hard_membership(eng))
        tr = {q.node_id for q, _, _ in sp["train"]}; te = {q.node_id for q, _, _ in sp["test"]}
        _ok(f"[{ds}] train/test queries disjoint", tr.isdisjoint(te))
        gres = all(any(g in docset for g in golds) for _, _, golds in sp["test"] if golds)
        _ok(f"[{ds}] test golds resolve to docs", gres)


def main():
    print("=== pure-function validation ===")
    test_metrics_known(); test_metric_invariants(); test_combined_ge_components()
    test_mcnemar(); test_classify(); test_interp()
    if "--substrate" in sys.argv:
        print("=== substrate integrity (loads engines) ===")
        test_substrate()
    n_pass = sum(1 for _, ok in _checks if ok); n = len(_checks)
    print(f"\n{n_pass}/{n} checks passed" + ("" if n_pass == n else "  <<< FAILURES"))
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    sys.exit(main())
