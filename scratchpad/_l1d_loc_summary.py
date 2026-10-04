"""Summary of the L1 development LOC runs (scratchpad/_l1d_loc.py): budget curves, strata, the A/B/C audit and the
'semantically invisible' gold analysis, read-only over results/L1_DEV/loc_<dataset>__<tag>.{json,npz}.

Usage: python scratchpad/_l1d_loc_summary.py <tag>  ->  results/L1_DEV/loc_SUMMARY__<tag>.json (write-once) + markdown on stdout.
DEVELOPMENT numbers (user ruling 2026-09-26): descriptive, no verdicts."""
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), os.path.join(REPO, "results", "L1_DEV"))
TAG = sys.argv[1]
DATASETS = ("metaqa", "squad", "musique")
PAIRED = "paired_vs_FLAT (gained = arm serves ALL gold, FLAT does not; McNemar p descriptive)"
AUDIT = "audit_ABC (FLAT@M misses vs the LOC order at M)"
SLOTK = "SLOT_C_at_M5000 (ruling-1 interface)"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def f4(x):
    return "%.4f" % x


def served_all(pos, M, gptr, ngold):
    cnt = np.add.reduceat((pos < M).astype(np.int64), gptr[:-1])
    return cnt == ngold


summ = {"tag": TAG, "status": "DEVELOPMENT (descriptive; not confirmatory)", "inputs": {}, "datasets": {}}
md = []
for ds in DATASETS:
    fj = os.path.join(OUT, "loc_%s__%s.json" % (ds, TAG))
    if not os.path.exists(fj):
        md.append("\n(%s: no record)" % ds)
        continue
    r = json.load(open(fj, encoding="utf-8"))
    fz = os.path.join(OUT, r["npz"]["path"])
    assert sha_file(fz) == r["npz"]["sha256"], "npz does not match its record"
    summ["inputs"][ds] = {"json": {"path": os.path.relpath(fj, REPO).replace("\\", "/"), "sha256": sha_file(fj)},
                          "npz": {"path": os.path.relpath(fz, REPO).replace("\\", "/"), "sha256": r["npz"]["sha256"]}}
    z = np.load(fz)
    gptr = z["gptr"]
    ngold = np.diff(gptr)
    posF = z["pos_FLAT"]
    Ms = [int(m) for m in r["FLAT"]]
    MMAX = max(Ms)
    D = {"N": r["N"], "n_rows": r["n_rows"], "n_gold_nodes": r["n_gold_nodes"], "cells": {}}
    md.append("\n### %s (N = %d, %d queries, %d gold nodes)\n" % (ds, r["N"], r["n_rows"], r["n_gold_nodes"]))
    for c in r["cells"]:
        cr = r["cells_result"][c]
        C = {"partition": cr["partition"]["tag"], "npart": cr["partition"]["npart"], "curve": {}, "strata": {}, "audit": {}, "invisible": {}}
        md.append("\n**cell %s** (partition %s, %d blocks, block size median %s; LOC order length median %s; gold in an activated block %s)\n" % (
            c, cr["partition"]["tag"], cr["partition"]["npart"], cr["partition"]["block_size"]["median"],
            cr["loc_gold_free"]["loc_order_length"]["median"], cr["loc_gold_free"]["gold_nodes_in_an_activated_block"]))
        md.append("| M | FLAT ALL | LOC ALL (+/−) | FLAT+LOC ALL (+/−) | FLAT M' matching LOC / FLAT+LOC | ANY F / L / F+L | FRAC F / L / F+L |")
        md.append("|---|---|---|---|---|---|---|")
        for M in Ms:
            k = str(M)
            F, L, FL = r["FLAT"][k], cr["arms"]["LOC"][k], cr["arms"]["FLAT+LOC"][k]
            C["curve"][k] = {"FLAT": {x: F[x] for x in ("ALL", "ANY", "FRAC")},
                             "LOC": {x: L[x] for x in ("ALL", "ANY", "FRAC")}, "FLAT+LOC": {x: FL[x] for x in ("ALL", "ANY", "FRAC")},
                             "LOC_paired": L[PAIRED], "FLAT+LOC_paired": FL[PAIRED],
                             "LOC_FLAT_budget_matching": L["FLAT_budget_matching_this_ALL"],
                             "FLAT+LOC_FLAT_budget_matching": FL["FLAT_budget_matching_this_ALL"]}
            md.append("| %d | %s | %s (+%d/−%d) | %s (+%d/−%d) | %d / %d | %s / %s / %s | %s / %s / %s |" % (
                M, f4(F["ALL"]), f4(L["ALL"]), L[PAIRED]["gained"], L[PAIRED]["lost"], f4(FL["ALL"]), FL[PAIRED]["gained"], FL[PAIRED]["lost"],
                L["FLAT_budget_matching_this_ALL"], FL["FLAT_budget_matching_this_ALL"],
                f4(F["ANY"]), f4(L["ANY"]), f4(FL["ANY"]), f4(F["FRAC"]), f4(L["FRAC"]), f4(FL["FRAC"])))
        sl = cr[SLOTK]
        C["slot_c"] = {k: {"ALL": v["ALL"], "paired": v[PAIRED]} for k, v in sl.items()}
        md.append("\nSLOT_C at M = %d (FLAT[:M−C] + C novel LOC nodes): %s  (FLAT@%d = %s)\n" % (
            MMAX, " · ".join("C=%s %s (+%d/−%d)" % (k, f4(v["ALL"]), v[PAIRED]["gained"], v[PAIRED]["lost"]) for k, v in sl.items()),
            MMAX, f4(r["FLAT"][str(MMAX)]["ALL"])))
        # strata tables: ALL by stratum x M for the three arms
        for sn in r["FLAT"][str(Ms[0])]["strata"]:
            keys = list(r["FLAT"][str(Ms[0])]["strata"][sn])
            C["strata"][sn] = {}
            md.append("| %s | n | " % sn + " | ".join("M=%d F / L / F+L" % M for M in Ms) + " |")
            md.append("|---|---|" + "---|" * len(Ms))
            for kk in keys:
                row = []
                C["strata"][sn][kk] = {}
                for M in Ms:
                    k = str(M)
                    a = r["FLAT"][k]["strata"][sn][kk]["ALL"]
                    b = cr["arms"]["LOC"][k]["strata"][sn][kk]["ALL"]
                    d = cr["arms"]["FLAT+LOC"][k]["strata"][sn][kk]["ALL"]
                    C["strata"][sn][kk][k] = {"FLAT": a, "LOC": b, "FLAT+LOC": d}
                    row.append("%.3f / %.3f / %.3f" % (a, b, d))
                md.append("| %s | %d | " % (kk, r["FLAT"][str(Ms[0])]["strata"][sn][kk]["n"]) + " | ".join(row) + " |")
            md.append("")
        # A/B/C audit
        md.append("| M | FLAT@M-missed gold nodes | LOC serves | Type A (block silent) | Type B (block too weak) | Type C (too deep in block) | FLAT+LOC serves |")
        md.append("|---|---|---|---|---|---|---|")
        for M in Ms:
            a = cr[AUDIT][str(M)]
            C["audit"][str(M)] = {"n_missed": a["n_FLAT@M_missed_gold_nodes"], "counts": a["counts"],
                                  "FLAT+LOC_serves": a["FLAT+LOC@M serves (of the FLAT@M misses)"],
                                  "LOC_serves_FLAT_rank_bins": a["FLAT_rank_bins_of_golds_LOC@M_serves"],
                                  "by_stratum": a["by_stratum"]}
            n = a["n_FLAT@M_missed_gold_nodes"]
            cnt = a["counts"]
            md.append("| %d | %d | %d | %d | %d | %d | %d |" % (M, n, cnt["LOC_SERVES"], cnt["TYPE_A"], cnt["TYPE_B"], cnt["TYPE_C"],
                                                         a["FLAT+LOC@M serves (of the FLAT@M misses)"]))
        # the audit split by stratum (hop, answer cardinality) at M = 1000 and the largest budget
        for M in (1000, MMAX):
            bs = cr[AUDIT][str(M)]["by_stratum"]
            if not bs:
                continue
            md.append("\nA/B/C of the FLAT@%d misses by stratum (gold nodes):\n" % M)
            md.append("| stratum | missed | LOC serves | Type A | Type B | Type C | FLAT+LOC serves |")
            md.append("|---|---|---|---|---|---|---|")
            for sn, e in bs.items():
                for kk, v in e.items():
                    md.append("| %s %s | %d | %d | %d | %d | %d | %d |" % (sn.replace("per_", ""), kk, v["n_missed_gold_nodes"], v["LOC_SERVES"],
                                                                         v["TYPE_A"], v["TYPE_B"], v["TYPE_C"], v["FLAT+LOC_serves"]))
        md.append("")
        # 'semantically invisible' gold nodes: FLAT rank >= MMAX (FLAT serves them at no budget of the curve)
        inv = posF >= MMAX
        posL, posFL = z["pos_LOC__" + c], z["pos_FLATLOC__" + c]
        lpos = z["lpos__" + c]
        C["invisible"] = {"definition": "gold nodes with FLAT rank >= %d (unserved by FLAT at every budget of the curve)" % MMAX,
                          "n": int(inv.sum()), "share_of_gold_nodes": round(float(inv.mean()), 4),
                          "in_an_activated_block": int((inv & (lpos >= 0)).sum()),
                          "LOC_serves_at_M": {str(M): int((inv & (posL < M)).sum()) for M in Ms},
                          "FLAT+LOC_serves_at_M": {str(M): int((inv & (posFL < M)).sum()) for M in Ms}}
        md.append("\nSemantically invisible gold nodes (FLAT rank ≥ %d): %d of %d (%.1f%%); in an activated block %d; LOC serves %s; FLAT+LOC serves %s\n" % (
            MMAX, int(inv.sum()), len(inv), 100.0 * inv.mean(), C["invisible"]["in_an_activated_block"],
            ", ".join("@%d %d" % (M, C["invisible"]["LOC_serves_at_M"][str(M)]) for M in Ms),
            ", ".join("@%d %d" % (M, C["invisible"]["FLAT+LOC_serves_at_M"][str(M)]) for M in Ms)))
        # per-hop and per-cardinality split of the invisible gold nodes (hop from the canonical query id: metaqa ':<h>hop:',
        # musique '<h>hop...'); served counts at the largest budget
        hq = [re.search(r"(\d)hop", q) for q in r["_row_query_ids"]]
        row_of_gold = np.repeat(np.arange(len(ngold)), ngold)
        groups = {}
        if all(hq):
            hop_row = np.array([int(m.group(1)) for m in hq], np.int64)
            for h in sorted(set(hop_row.tolist())):
                groups["hop%d" % h] = hop_row[row_of_gold] == h
        for nm, lo, hi in (("1", 1, 1), ("2-4", 2, 4), ("5-10", 5, 10), ("11+", 11, 1 << 40)):
            groups["n_gold " + nm] = (ngold[row_of_gold] >= lo) & (ngold[row_of_gold] <= hi)
        C["invisible_by_group"] = {}
        for g, m in groups.items():
            if not (inv & m).any():
                continue
            C["invisible_by_group"][g] = {"gold_nodes": int(m.sum()), "invisible": int((inv & m).sum()),
                                          "LOC_serves_at_M": {str(M): int((inv & m & (posL < M)).sum()) for M in Ms},
                                          "FLAT+LOC_serves_at_M": {str(M): int((inv & m & (posFL < M)).sum()) for M in Ms}}
        if C["invisible_by_group"]:
            md.append("| group | gold nodes | invisible (FLAT rank ≥ %d) | LOC serves @1000 / @%d | FLAT+LOC serves @1000 / @%d |" % (MMAX, MMAX, MMAX))
            md.append("|---|---|---|---|---|")
            for g, e in C["invisible_by_group"].items():
                md.append("| %s | %d | %d | %d / %d | %d / %d |" % (g, e["gold_nodes"], e["invisible"], e["LOC_serves_at_M"]["1000"],
                                                                 e["LOC_serves_at_M"][str(MMAX)], e["FLAT+LOC_serves_at_M"]["1000"],
                                                                 e["FLAT+LOC_serves_at_M"][str(MMAX)]))
            md.append("")
        C["latency_ms"] = cr["latency_ms"]
        C["structure_bytes"] = cr["structure_bytes"]
        D["cells"][c] = C
    D["latency_ms_flat"] = r["latency_ms (this exhaustive CPU implementation, per query)"]
    D["index_bytes"] = r["index_bytes"]
    D["served_list_agreement"] = r["served_list_agreement"]
    D["peak_rss_mb"] = r["process_peak_rss_mb"]
    D["seconds"] = r["seconds"]
    md.append("Latency per query (exhaustive CPU implementation): FLAT products %s ms, FLAT RRF sort %s ms; " % (
        D["latency_ms_flat"]["flat_products_amortized (dense + SPLADE over all N)"]["mean_ms"], D["latency_ms_flat"]["flat_rrf (full sort)"]["mean_ms"])
              + "; ".join("%s LOC %s ms, fusion %s ms" % (c, D["cells"][c]["latency_ms"]["loc (activation + LOC order)"]["mean_ms"],
                                                          D["cells"][c]["latency_ms"]["fusion (f + lexsort over N)"]["mean_ms"]) for c in r["cells"])
              + ". Index: dense %.1f MB, SPLADE %.1f MB; LOC structure %s. Peak RSS %s MB, %s s." % (
                  D["index_bytes"]["dense_fp16 (N x dim x 2)"] / 1e6, D["index_bytes"]["splade_csr (nnz x 8 + (N + 1) x 8)"] / 1e6,
                  ", ".join("%s %.2f MB" % (c, sum(v for k, v in D["cells"][c]["structure_bytes"].items() if k != "partition_file") / 1e6) for c in r["cells"]),
                  D["peak_rss_mb"], D["seconds"]))
    summ["datasets"][ds] = D
fo = os.path.join(OUT, "loc_SUMMARY__%s.json" % TAG)
assert not os.path.exists(fo), "write-once: %s exists" % fo
with open(fo, "w", encoding="utf-8") as f:
    json.dump(summ, f, indent=1, ensure_ascii=True)
print("\n".join(md))
print("\n-> %s sha256 %s" % (fo, sha_file(fo)))
