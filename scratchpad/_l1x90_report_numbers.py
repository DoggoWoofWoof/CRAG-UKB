"""Post-confirmation numbers for the report: full dev (A+B) per dataset, residual categories (metaqa),
oracle ceilings, exposure.  Writes results/L1_P90_EXPLOIT/REPORT_NUMBERS.json."""
import collections
import json
import os

import numpy as np

import _l1x90_core as X
import _l1x90_relwalk as RW
import _l1x90_seeds as SE
import _l1x90_typed as TY
import _l1x90_universal as UN

out = {}
for name in ("metaqa", "metaqa_phg", "squad", "musique"):
    C = X.Cache(name)
    T_rank = C.base_rank.astype(np.int64)
    base_sel = X.fuse(C, T_rank, T_rank, "T")
    r0, base_all, base_any = X.evaluate(C, base_sel, "BASE")
    sel, mode, info = UN.universal_select(C, "G2", cache_tag=name)
    r, allv, anyv = X.evaluate(C, sel, "G2")
    ngb = np.array([len(g) for g in C.gb])
    feas = ngb <= X.P_MAIN
    pt = X.rank_pos(T_rank, C.npart)
    reach_t = np.array([all(pt[i, p] < 10 ** 6 for p in C.gb[i]) for i in range(C.nq)])
    hops = sorted(set(int(h) for h in C.hops if h >= 0))

    def by_hop(v):
        return {("hop%d" % h): float(v[C.hops == h].mean()) for h in hops} if hops else {}

    g, l, p = X.mcnemar(base_all, allv)
    e = {"n": int(C.nq), "n_A": int((C.split == "A").sum()), "n_B": int((C.split == "B").sum()), "npart": int(C.npart),
         "BASE": {"ALL": float(base_all.mean()), "ANY": float(base_any.mean()), "per_hop": by_hop(base_all), "scope_nodes": r0["scope_nodes"]},
         "G2": {"ALL": float(allv.mean()), "ANY": float(anyv.mean()), "per_hop": by_hop(allv), "scope_nodes": r["scope_nodes"],
                "gained": g, "lost": l, "p_mcnemar": p, "typed_rate": float(mode.mean()), "info": info},
         "A": {"BASE": r0["ALL_split"]["A"]["ALL"], "G2": r["ALL_split"]["A"]["ALL"]},
         "B": {"BASE": r0["ALL_split"]["B"]["ALL"], "G2": r["ALL_split"]["B"]["ALL"]},
         "oracle_feasible_at_P50": {"ALL": float(feas.mean()), "per_hop": by_hop(feas)},
         "text_reach_all_golds_within_200_ranked_blocks": float(reach_t.mean()),
         "gold_blocks_per_query": {"mean": float(ngb.mean()), "p90": float(np.percentile(ngb, 90)), "max": int(ngb.max()), "gt50": float((ngb > 50).mean())}}
    if name in ("metaqa", "metaqa_phg"):
        T = TY.Typed(C)
        Q = TY.questions_of(C)
        ch = UN.build_channels(C, T, Q, cache_tag=name)
        M = ch["seed"] + ch["ans"] + UN.EPS * (ch["oth"] + ch["unt"])
        S_rank = X.rank_from_scores(M)
        ps = X.rank_pos(S_rank, C.npart)
        qt, te = {}, {}
        with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "queries", "dev.jsonl"), encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                qt[o["query_id"]] = o["qtype"]
                te[o["query_id"]] = o.get("topic_entity_node_id")
        nid = {}
        with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "nodes.jsonl"), encoding="utf-8") as f:
            for k, line in enumerate(f):
                nid[json.loads(line)["node_id"]] = k
        REL = {"director": "directed_by", "writer": "written_by", "actor": "starred_actors", "genre": "has_genre", "year": "release_year",
               "language": "in_language", "tags": "has_tags", "imdbrating": "has_imdb_rating", "tag": "has_tags"}
        cat = collections.defaultdict(collections.Counter)
        seed_ok = sched_ok = 0
        for i in range(C.nq):
            topic = nid.get(te[C.qids[i]])
            sd = set(int(x) for x in ch["seeds"][i] if x >= 0)
            s_ok = topic in sd
            seed_ok += s_ok
            got = ch["orders"][i].split(",") if ch["orders"][i] else []
            parts = [REL[p_] for p_ in qt[C.qids[i]].split("_to_") if p_ in REL]
            exp = []
            for p_ in parts:
                if p_ not in exp:
                    exp.append(p_)
            sc_ok = got == exp
            sched_ok += sc_ok
            if allv[i]:
                continue
            h = int(C.hops[i])
            missed = [p_ for p_ in C.gb[i] if ps[i, p_] >= X.P_MAIN]
            if not s_ok:
                c = "seed_not_found"
            elif not sc_ok:
                c = "schedule_mismatch(lexically unmatchable relation words)"
            elif ngb[i] > X.P_MAIN - 1:
                c = "infeasible(>49 gold blocks)"
            elif all(ch["ans"][i, p_] == 0 for p_ in missed):
                c = "gold_block_without_answer_mass"
            else:
                c = "ranking_within_capacity"
            cat[h][c] += 1
        e["diagnostics_full_dev"] = {"topic_entity_in_seeds": seed_ok / C.nq, "schedule_exact_vs_qtype": sched_ok / C.nq,
                                    "misses_by_hop": {("hop%d" % h): dict(cat[h]) for h in sorted(cat)}, "n_misses": int(C.nq - allv.sum())}
    out[name] = e
    print("%-11s n=%d BASE ALL %.4f ANY %.4f | G2 ALL %.4f ANY %.4f (+%d/-%d p=%.2g) per hop %s | feasible@50 %.4f | scope nodes BASE %s G2 %s" % (
        name, C.nq, e["BASE"]["ALL"], e["BASE"]["ANY"], e["G2"]["ALL"], e["G2"]["ANY"], g, l, p,
        {k: round(v, 3) for k, v in e["G2"]["per_hop"].items()}, e["oracle_feasible_at_P50"]["ALL"], r0["scope_nodes"], r["scope_nodes"]), flush=True)
    if "diagnostics_full_dev" in e:
        print("   ", json.dumps(e["diagnostics_full_dev"]))
X.wj(os.path.join(X.OUT, "REPORT_NUMBERS.json"), out)
