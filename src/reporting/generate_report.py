"""
Benchmark/ablation report generator (phase 9).
==============================================
Reads the UKB result store (data/ukb_storage/{ds}/results/{L1,L2,L3,cross}/) and
emits a single queryable markdown: per-dataset champion + cross-dataset summary
tables for L1 (champion / HNM verdict / loss), L1+L3 recall, L3 (reachability /
traversal / latency), and L2 seed-minimization. Robust to missing artifacts, so
it works incrementally as phases land. Rebuilds the manifest first.

Writes data/ukb_storage/_index/BENCHMARK.md.
"""
import os
import json
import logging

from src.pipeline.ukb_results import UKB, INDEX_DIR, DATASETS, build_index

log = logging.getLogger("reporting.generate_report")


def _load(ds, level, artifact):
    p = os.path.join(UKB, ds, "results", level, artifact)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None
    return None


def _present_datasets():
    return [d for d in dict.fromkeys(DATASETS) if os.path.isdir(os.path.join(UKB, d, "results"))]


def _hnm_verdict(d):
    if not d or "results" not in d:
        return None
    r = d["results"]
    def f50(k): return (r.get(k) or {}).get("full_coverage@50")
    ks = [k for k in r if k.startswith("hnk_")]
    allk = "hnk_all"
    if allk not in r:
        return None
    others = [k for k in ks if k != allk]
    all_best = all((f50(allk) or 0) >= (f50(k) or 0) for k in others)
    return f"full-softmax best ({f50(allk)})" + ("; monotonic ✓" if all_best else "; NON-monotonic")


def _lines():
    L = ["# CRAG clean benchmark — L1 → L2 → L3", "",
         "_Generated from the UKB result store (`data/ukb_storage/{dataset}/results/`)._", ""]
    dss = _present_datasets()

    # ---- L1 champion table ----
    L += ["## L1 — per-dataset champion (pool-matched @10% pool)", "",
          "| dataset | champion | loss | FullCov | gtRecall | HNM verdict |", "|---|---|---|---|---|---|"]
    for ds in dss:
        ch = _load(ds, "L1", "champion.json"); hn = _load(ds, "L1", "hnm_sweep.json")
        c = (ch or {}).get("champion_10pct", {})
        L.append(f"| {ds} | {c.get('config','—')} | {c.get('loss','—')} | "
                 f"{c.get('fullcov_pm','—')} | {c.get('gt_recall_pm','—')} | {_hnm_verdict(hn) or '—'} |")
    L.append("")

    # ---- L1+L3 recall (cross) ----
    L += ["## L1+L3 recall (real router front-end, gt_recall@200)", "",
          "| dataset | L1 dense | L1 1hop | L1 ppr | ideal ppr | pool ceiling |", "|---|---|---|---|---|---|"]
    for ds in dss:
        d = _load(ds, "cross", "l1l3.json")
        if not d or "results" not in d:
            L.append(f"| {ds} | — | — | — | — | — |"); continue
        R = d["results"]; g = lambda k: (R.get(k, {}).get("gt_recall", {}) or {}).get("gt_recall@200", "—")
        ceil = (R.get("L1_pool_ceiling", {}) or {}).get("gt_recall_pool", "—")
        L.append(f"| {ds} | {g('L1:dense')} | {g('L1:1hop')} | {g('L1:ppr')} | {g('ideal:ppr')} | {ceil} |")
    L.append("")

    # ---- L3 reachability + traversal ----
    L += ["## L3 — reachability ceiling, traversal, latency", "",
          "| dataset | reach@2hop | frontier@2hop | traverse recall@200 | avg nodes | median ms |",
          "|---|---|---|---|---|---|"]
    for ds in dss:
        rc = _load(ds, "L3", "reachability.json"); tv = _load(ds, "L3", "traverse.json")
        lt = _load(ds, "L3", "latency.json")
        r2 = (rc or {}).get("reachability_pct", {}).get("h2", "—")
        f2 = (rc or {}).get("frontier_pct_of_corpus", {}).get("h2", "—")
        tr = (tv or {}).get("gt_recall", {}).get("@200", "—")
        nu = (tv or {}).get("avg_nodes_expanded", "—")
        ms = (tv or {}).get("median_latency_ms") or (lt or {}).get("methods", {}).get("ppr", {}).get("median_ms_per_query", "—")
        L.append(f"| {ds} | {r2} | {f2} | {tr} | {nu} | {ms} |")
    L.append("")

    # ---- L2 seed minimization ----
    L += ["## L2 — minimal seeds (cross-encoder vs dense)", "",
          "| dataset | dense min-seeds@target | cross-encoder min-seeds@target |", "|---|---|---|"]
    for ds in dss:
        d = _load(ds, "L2", "seed_selection.json")
        if not d:
            L.append(f"| {ds} | — | — |"); continue
        L.append(f"| {ds} | {d.get('dense',{}).get('min_seeds_for_target','—')} | "
                 f"{d.get('cross_encoder',{}).get('min_seeds_for_target','—')} |")
    L.append("")

    # ---- L3 traceability examples ----
    L += ["## L3 traceability — example evidence paths", ""]
    for ds in dss:
        tv = _load(ds, "L3", "traverse.json")
        for t in (tv or {}).get("example_traces", [])[:2]:
            L.append(f"- **{ds}** ({t.get('stop')}, {t.get('nodes')} nodes): "
                     f"`{t.get('question','')}` → path `{' → '.join(t.get('example_path_to_gold', []))}`")
    L.append("")
    return L


def run():
    build_index()
    os.makedirs(INDEX_DIR, exist_ok=True)
    path = os.path.join(INDEX_DIR, "BENCHMARK.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines()))
    log.info(f"wrote {path}")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
