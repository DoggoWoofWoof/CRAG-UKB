"""
Universal (dataset-agnostic) fusion gate on [overlap, dense_conf].
==================================================================
The per-dataset overlap-gate fixes hotpot, but overlap ALONE can't be made
universal: hotpot overlap (0.33) ~ musique (0.30) yet they want opposite rel
weights, because overlap measures divergence, not whether divergence helps. The
missing axis is dense's absolute strength -> add dense_conf (top-1 cosine).
Hypothesis: a single rule on the COMBINED signal separates them (drop rel only
when dense is strong AND rel is redundant).

Reads the CACHED orders from l1_fusion (no retraining — instant). Fits, on POOLED
train (subsampled, maximizing the MEAN per-dataset recall@100 so big datasets don't
dominate), one universal rule:
    combined(q) = overlap(q) + lam * dense_conf(q)
    rel_weight(q) = clip(a * (b - combined(q)), lo, hi)
Compares, per dataset + mean: equal_rrf | per_dataset_overlap_gate | UNIVERSAL gate.
The question: does the universal 2-signal gate match the per-dataset gates? If yes,
we have a dataset-agnostic fusion rule. Writes _index/l1_universal_gate_summary.json.
"""
import os
import json
import logging
import argparse
import itertools

import numpy as np

from src.experiments.l1_fusion import CACHE_DIR, _overlap, _gate
from src.experiments.l1_ablate import _ranks, _rrf_fuse, _recall


log = logging.getLogger("experiments.l1_universal_gate")


def load(ds, limit=20000):
    npz = os.path.join(CACHE_DIR, f"{ds}_L{limit}.npz")
    gj = os.path.join(CACHE_DIR, f"{ds}_L{limit}_gold.json")
    z = np.load(npz); g = json.load(open(gj, encoding="utf-8"))
    return {k: z[k] for k in z.files} | {"gold_tr": g["tr"], "gold_te": g["te"]}


def _combined(overlap, conf, lam):
    return overlap + lam * conf


def _subidx(n, nsub=2000):
    return np.linspace(0, n - 1, min(n, nsub)).astype(int)


def run(datasets, limit=20000):
    S = {}
    for ds in datasets:
        c = load(ds, limit)
        ov_tr, ov_te = _overlap(c["d_tr"], c["h_tr"]), _overlap(c["d_te"], c["h_te"])
        S[ds] = {
            "ov_tr": ov_tr, "conf_tr": c["conf_tr"], "gold_tr": c["gold_tr"],
            "ov_te": ov_te, "conf_te": c["conf_te"], "gold_te": c["gold_te"],
            "dmap_tr": _ranks(c["d_tr"]), "hmap_tr": _ranks(c["h_tr"]), "tmap_tr": _ranks(c["t_tr"]),
            "dmap": _ranks(c["d_te"]), "hmap": _ranks(c["h_te"]), "tmap": _ranks(c["t_te"]),
        }
        log.info(f"[{ds}] loaded (overlap~{ov_te.mean():.3f}, conf~{c['conf_te'].mean():.3f})")

    def fused_recall(dm, hm, tm, w, gold, sub=None):
        if sub is not None:
            dm = [dm[i] for i in sub]; hm = [hm[i] for i in sub]; tm = [tm[i] for i in sub]
            gold = [gold[i] for i in sub]; w = w[sub]
        return _recall(_rrf_fuse([dm, hm, tm], [np.ones_like(w), w, w]), gold, budgets=[100])[100]

    # ---- per-dataset overlap-only gate (baseline to beat/match)
    perds = {}
    for ds, s in S.items():
        sub = _subidx(len(s["ov_tr"]))
        best, ab = -1, (3.0, 0.5)
        for a in (1., 2., 3., 5.):
            for b in (0.2, 0.4, 0.6, 0.8, 0.9):
                r = fused_recall(s["dmap_tr"], s["hmap_tr"], s["tmap_tr"], _gate(s["ov_tr"], a, b), s["gold_tr"], sub)
                if r > best:
                    best, ab = r, (a, b)
        perds[ds] = ab

    # ---- UNIVERSAL gate on [overlap, conf]: one (a,b,lam) maximizing MEAN per-dataset train recall@100
    grid = list(itertools.product((2., 3., 5.), (0.4, 0.6, 0.8, 1.0, 1.2), (0.0, 0.5, 1.0, 2.0)))
    subs = {ds: _subidx(len(S[ds]["ov_tr"])) for ds in S}
    best_u, uni = -1, (3.0, 0.8, 1.0)
    for a, b, lam in grid:
        recs = []
        for ds, s in S.items():
            w = _gate(_combined(s["ov_tr"], s["conf_tr"], lam), a, b)
            recs.append(fused_recall(s["dmap_tr"], s["hmap_tr"], s["tmap_tr"], w, s["gold_tr"], subs[ds]))
        mean = float(np.mean(recs))
        if mean > best_u:
            best_u, uni = mean, (a, b, lam)
    a_u, b_u, lam_u = uni

    # ---- evaluate on TEST
    out = {"datasets": list(S), "universal_abl": [a_u, b_u, lam_u], "per_dataset_ab": perds, "per_dataset": {}}
    for ds, s in S.items():
        equal = _recall(_rrf_fuse([s["dmap"], s["hmap"], s["tmap"]], [1., 1., 1.]), s["gold_te"])[100]
        w_pd = _gate(s["ov_te"], *perds[ds])
        pdg = _recall(_rrf_fuse([s["dmap"], s["hmap"], s["tmap"]], [np.ones_like(w_pd), w_pd, w_pd]), s["gold_te"])[100]
        w_u = _gate(_combined(s["ov_te"], s["conf_te"], lam_u), a_u, b_u)
        ug = _recall(_rrf_fuse([s["dmap"], s["hmap"], s["tmap"]], [np.ones_like(w_u), w_u, w_u]), s["gold_te"])[100]
        out["per_dataset"][ds] = {"equal_rrf": equal, "per_dataset_gate": round(pdg, 2),
                                  "universal_gate": round(ug, 2), "univ_relw_mean": round(float(w_u.mean()), 3),
                                  "univ_vs_equal": round(ug - equal, 2), "univ_vs_perds": round(ug - pdg, 2)}
        log.info(f"[{ds}] equal {equal} | perds_gate {pdg:.2f} | UNIVERSAL {ug:.2f} "
                 f"(relw~{w_u.mean():.2f}, vs_equal {ug-equal:+.2f}, vs_perds {ug-pdg:+.2f})")
    pd = out["per_dataset"]
    out["mean"] = {k: round(float(np.mean([pd[ds][k] for ds in pd])), 2)
                   for k in ("equal_rrf", "per_dataset_gate", "universal_gate")}
    os.makedirs(os.path.join("data", "ukb_storage", "_index"), exist_ok=True)
    with open(os.path.join("data", "ukb_storage", "_index", "l1_universal_gate_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"UNIVERSAL GATE (a={a_u},b={b_u},lam={lam_u}): mean equal {out['mean']['equal_rrf']} | "
             f"perds {out['mean']['per_dataset_gate']} | UNIVERSAL {out['mean']['universal_gate']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Universal 2-signal fusion gate (cached orders).")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean", "metaqa"])
    p.add_argument("--limit", type=int, default=20000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("===== L1 UNIVERSAL GATE ([overlap, dense_conf]) =====")
    run(a.datasets, a.limit)


if __name__ == "__main__":
    main()
