"""POST-HOC DESCRIPTIVE ONLY (labelled; no decision, no rule): the STRUCT_VERTEXCUT_V1_TRANSFER primary / matched cells of a dataset
broken down by the number of gold nodes per query (musique: 2 / 3 / 4 gold paragraphs = its 2 / 3 / 4-hop questions; the served cache
carries no hop labels).  Recomputes the same deterministic pipeline as _l1c_vcut_transfer.py (functions imported unchanged) and asserts
that the pooled numbers equal the recorded ones before writing the breakdown.

    python -u scratchpad/_l1c_vcut_transfer_posthoc.py <ds>  -> results/L1_COVPART/vcut_transfer_A_<ds>__posthoc_by_gold_count.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402
import _l1g_core as G  # noqa: E402
import _l1c_vcut_lib as V  # noqa: E402
import _l1c_vcut_replay as R  # noqa: E402
import _l1c_vcut_transfer as T  # noqa: E402
from src.l1_canonical.adapter import CanonicalDataset  # noqa: E402

X = S.X
K_LOCK, P_MAIN = S.K_LOCK, S.P_MAIN
OUT = V.OUT
log = S.log
t0 = time.time()
ds = sys.argv[1]
gate_label, gate_cache, _ = T.GATE[ds]
rec = json.load(open(os.path.join(OUT, "vcut_transfer_A_%s.json" % ds), encoding="utf-8"))
k = rec["representation"]["k"]
cap_p = os.path.join(X.REPO, rec["substrate"]["path"])
assert R.sha_file(cap_p) == rec["substrate"]["sha256"]
D = G.Data(gate_cache, dense_fp32=False)
mA = D.C.split == "A"
Gr = V.Graph(ds)
z = np.load(cap_p).astype(np.int64)
mem_all0, lam0, home0, ties, sizes0, blocks_csc0 = R.memberships(Gr, z, k)
deg0 = Gr.deg == 0
src, dst, w, ent = CanonicalDataset(ds).family("knn")
home_d0, anchored, fallback, sizes_after, d0 = T.place_degree0(Gr, home0, sizes0, k, (src, dst, w))
mem_all, home, blocks_csc, sizes = T.extend_membership(mem_all0, home0, home_d0, deg0, Gr.N, k)
mem_vc = R.vote_table(mem_all, home, Gr.hub)
base = D.base_all.astype(np.int8)
exp = np.array([int(D.sizes[np.asarray(s, np.int64)].sum()) for s in D.base_sel])
R_vc = R.fused_rank(D, mem_vc, k)
sel50 = [R_vc[i, :P_MAIN] for i in range(D.nq)]
all50, any50 = R.served(D, sel50, mem_all)
u50, _ = R.unique_exposure(sel50, blocks_csc, Gr.N)
L_le, U_le, L_ge, U_ge = R.matched_prefixes(R_vc, blocks_csc, Gr.N, exp)
all_le, _ = R.served(D, [R_vc[i, :L_le[i]] for i in range(D.nq)], mem_all)
# ---- the pooled numbers must equal the record
c = rec["cells"]
assert abs(float(all50[mA].mean()) - c["VCUT_P50"]["ALL"]["all"]) < 1e-4 and abs(float(all_le[mA].mean()) - c["VCUT_MATCHED_LE"]["ALL"]["all"]) < 1e-4
assert abs(float(base[mA].mean()) - c[gate_label]["ALL"]["all"]) < 1e-4 and abs(float(u50[mA].mean()) - c["VCUT_P50"]["exposure_unique_A"]["mean"]) < 0.01
# ---- reach for the primary cell
ev = R.evidence(D, mem_vc, k)
ptr, flat = mem_all
reached = np.zeros(D.nq, bool)
n_gold = np.zeros(D.nq, np.int64)
for i, g in enumerate(D.C.gold_nodes):
    n_gold[i] = len(g)
    reached[i] = all(bool(ev[i, flat[ptr[x]:ptr[x + 1]]].any()) for x in g) if len(g) else False
masks = {"all": mA}
for n in sorted(set(int(x) for x in n_gold[mA])):
    masks["gold_%d" % n] = mA & (n_gold == n)
out = {"RECORD": "STRUCT_VERTEXCUT_V1_TRANSFER_POSTHOC_BY_GOLD_COUNT", "status": "POST-HOC DESCRIPTIVE ONLY -- no decision cell, no rule; the pooled numbers were asserted equal to the recorded ones",
       "dataset": ds, "gate": gate_label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "transfer_record": R.pin(os.path.join(OUT, "vcut_transfer_A_%s.json" % ds)),
       "note": "musique's served cache carries no hop labels; the number of gold paragraphs (2 / 3 / 4) is its 2 / 3 / 4-hop question type",
       "by_gold_count": {}}
for nm, m in masks.items():
    fail = (all50 == 0) & m
    out["by_gold_count"][nm] = {"n": int(m.sum()), "gate_ALL": round(float(base[m].mean()), 4), "VCUT_P50_ALL": round(float(all50[m].mean()), 4), "VCUT_P50_ANY": round(float(any50[m].mean()), 4),
                                "P50_vs_gate": R.compare(base, all50, {nm: m})[nm], "VCUT_MATCHED_LE_ALL": round(float(all_le[m].mean()), 4), "MATCHED_LE_vs_gate": R.compare(base, all_le, {nm: m})[nm],
                                "unique_exposure_P50_mean": round(float(u50[m].mean()), 1), "hard_exposure_mean": round(float(exp[m].mean()), 1),
                                "matched_blocks_mean": round(float(L_le[m].mean()), 1),
                                "P50_failures": int(fail.sum()), "UNREACHED": int((fail & ~reached).sum()), "REACHED_WEAK": int((fail & reached).sum()), "reached_rate": round(float(reached[m].mean()), 4)}
out["code"] = R.pin(os.path.abspath(__file__))
out["seconds"] = round(time.time() - t0, 1)
fp = os.path.join(OUT, "vcut_transfer_A_%s__posthoc_by_gold_count.json" % ds)
S.wj(fp, out)
for nm, v in out["by_gold_count"].items():
    log("%s %s n %d: gate %.4f | P50 %.4f (+%d/-%d p=%.3g; unique %.0f vs hard %.0f) | MATCHED_LE %.4f (+%d/-%d p=%.3g; %.1f blocks) | failures %d = UNREACHED %d + WEAK %d" % (
        ds, nm, v["n"], v["gate_ALL"], v["VCUT_P50_ALL"], v["P50_vs_gate"]["gained"], v["P50_vs_gate"]["lost"], v["P50_vs_gate"]["p"], v["unique_exposure_P50_mean"], v["hard_exposure_mean"],
        v["VCUT_MATCHED_LE_ALL"], v["MATCHED_LE_vs_gate"]["gained"], v["MATCHED_LE_vs_gate"]["lost"], v["MATCHED_LE_vs_gate"]["p"], v["matched_blocks_mean"], v["P50_failures"], v["UNREACHED"], v["REACHED_WEAK"]))
log("-> %s (%.0fs)" % (os.path.relpath(fp, X.REPO), time.time() - t0))
