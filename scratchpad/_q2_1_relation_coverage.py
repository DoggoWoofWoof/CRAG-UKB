"""Q2.1 — EXACT RELATION EXPERT CONTROL (confirmatory, CPU-only, no 197M-row corpus, no Qwen for coverage).

The relation expert (l2_relation) fires on a candidate d ONLY IF title(d) is a title-mention target of the
query's topics: relation_mask(d) = [ title(d) in query_edges(M, q) ], where query_edges requires a sentence in
a topic entity's CONTENT to literally NAME title(d) (the `_mentions` gate). If mask=0 the relation RRF
contribution is EXACTLY 0 (ABSTAIN) — so relation can only *admit* a recovered gold if it FIRES on it.

This script answers the first-order Q2.1 question WITHOUT any encoder pass:
  For MetaQA VAL recovered golds (gold, hard-partition OUTSIDE the selected top-50, reachable by Track-A
  geometry at M), what fraction has title in query_edges (i.e. the exact relation expert FIRES)?
  Baseline: relation fire-rate on ALL in-scope golds (structural coverage of the expert on this KB corpus).

If the recovered-gold fire-rate is ~0, relation CANNOT admit them (abstains) and
CURRENT_EXISTING_EXPERT_SET_LACKS_RECOVERY_SIGNAL = YES — established without computing the cosine, because a
gold on which the expert abstains gets RRF contribution 0 and is never ranked above candidates that do fire.
(If the fire-rate were non-trivial, the follow-up cosine/rank pass would be needed — this gates that.)

Everything is inference-safe geometry + deterministic title matching; gold identity used ONLY to LABEL which
added rows are golds for the diagnostic (not fed to any scorer). TEST never touched.
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad")
import numpy as np
import g2_l1_geo as G
import l2_relation as RL
import _q2_eval as EV

DS = "metaqa"; SPLIT = "val"
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)


def main():
    Msweep = [int(x) for x in os.environ.get("Q21_M", "256").split(",")]
    subset = int(os.environ.get("Q21_SUBSET", "0"))     # per-hop cap; 0 = full valid val
    outp = os.environ.get("Q21_OUT", "results/GENERALIZATION/_q2_1_relation_coverage.json")

    A = G.load_artifacts(); d2i = A["doc_id_to_idx"]; hard = A["hard"]; mem_idx = A["mem_idx"]; npart = A["npart"]
    j = A["j"]; ids_g = j["ids"]; golds_g = j["golds"]; hops_g = j["hops"]
    val_rows = j["split_indices"]["val"]                 # row_all indices for the val split
    log(f"artifacts: N={A['N']} npart={npart} val={len(val_rows)}")

    # master + question text for relation query_edges
    from src.pipeline.standardizer import load_nodes
    qtext = {n.node_id: n.content for n in load_nodes(f"data/processed/master_nodes_{DS}.json")
             if n.metadata.get("type") == "question"}
    Mrel = RL.load_master(DS); log("relation master loaded")

    adj, sinfo = G.build_structural_adj(d2i); log(f"structural adj edges={sinfo['n_edges']}")
    nodes = np.load(f"{G.BASE}nodes.npy"); Xn = nodes / (np.linalg.norm(nodes, axis=1, keepdims=True) + 1e-9)
    qall = np.load(f"{G.BASE}queries_all.npy", mmap_mode="r")
    dense_all = np.load(f"{G.BASE}dense_top200_all.npy", mmap_mode="r")
    splade_all = np.load(f"{G.BASE}splade_top200_all.npy", mmap_mode="r")
    title_by_row = Mrel["title_by_row"]

    # choose eval rows (row_all space), grouped by hop
    by_hop = {1: [], 2: [], 3: []}
    for ra in val_rows:
        h = int(hops_g[ra]);
        if h in by_hop: by_hop[h].append(int(ra))
    eval_rows = []
    for h in (1, 2, 3):
        rs = by_hop[h]
        eval_rows += (rs[:subset] if subset > 0 else rs)
    log(f"eval rows: {len(eval_rows)} (hop split { {h: len(by_hop[h]) for h in by_hop} }; subset={subset})")

    # P50 selected partitions for all eval rows (batched fused router)
    K = G.K_LOCK; P = G.P_MAIN
    def sel_partitions(rows_batch):
        dK = np.stack([np.asarray(dense_all[i][:K]) for i in rows_batch]).astype(np.int64)
        sK = np.stack([np.asarray(splade_all[i][:K]) for i in rows_batch]).astype(np.int64)
        rank = G.fused_ranking(dK, sK, mem_idx, npart, K)
        return [set(int(x) for x in rank[qq][:P]) for qq in range(len(rows_batch))]

    res = {}
    for M in Msweep:
        agg = {h: {"n_recovered": 0, "rel_fires_recovered": 0,
                   "n_inscope_gold": 0, "rel_fires_inscope_gold": 0,
                   "n_q": 0, "example_fires": []} for h in (1, 2, 3)}
        BS = 256
        for b0 in range(0, len(eval_rows), BS):
            batch = eval_rows[b0:b0 + BS]
            sels = sel_partitions(batch)
            for k, ra in enumerate(batch):
                h = int(hops_g[ra]); sel = sels[k]
                q = qtext.get(ids_g[ra], "")
                seeds = G.resolve_seeds(q, d2i)
                if not seeds:
                    continue
                agg[h]["n_q"] += 1
                # geometry expansion set (rows), inference-safe — SAME residual + directional beam as Q2
                r_q = EV.residual(np.asarray(qall[ra], np.float64), seeds, Xn)
                exp = set(EV.expand_dir(seeds, r_q, adj, M, Xn))
                # golds
                gold_rows = [d2i[g] for g in golds_g[ra] if g in d2i]
                # query_edges titles (once per query)
                qedges = RL.query_edges(Mrel, q)          # {title: sentence}
                fires_titles = set(qedges.keys())
                for gr in gold_rows:
                    in_p50 = int(hard[gr]) in sel
                    ti = title_by_row.get(int(gr), "")
                    rel_fire = ti in fires_titles
                    if in_p50:
                        agg[h]["n_inscope_gold"] += 1
                        agg[h]["rel_fires_inscope_gold"] += int(rel_fire)
                    else:
                        # recovered = out-of-P50 gold that geometry co-scopes
                        if gr in exp:
                            agg[h]["n_recovered"] += 1
                            agg[h]["rel_fires_recovered"] += int(rel_fire)
                            if rel_fire and len(agg[h]["example_fires"]) < 5:
                                agg[h]["example_fires"].append({"gold_title": ti,
                                    "connecting_sentence": qedges[ti][:160], "hop": h})
            if (b0 // BS) % 10 == 0:
                log(f"  M={M} processed {b0+len(batch)}/{len(eval_rows)}")
        out = {}
        for h in (1, 2, 3):
            d = agg[h]; nr = max(d["n_recovered"], 1); ng = max(d["n_inscope_gold"], 1)
            out[str(h)] = {
                "n_queries_with_seed": d["n_q"],
                "n_recovered_golds": d["n_recovered"],
                "relation_FIRES_on_recovered": d["rel_fires_recovered"],
                "frac_recovered_relation_fires": round(d["rel_fires_recovered"] / nr, 4),
                "n_inscope_golds": d["n_inscope_gold"],
                "relation_FIRES_on_inscope_gold": d["rel_fires_inscope_gold"],
                "frac_inscope_gold_relation_fires": round(d["rel_fires_inscope_gold"] / ng, 4),
                "example_fires": d["example_fires"],
            }
        res[f"M{M}"] = out
        log(f"M={M} DONE: " + json.dumps({h: {"recov": out[h]["n_recovered_golds"],
             "rel_fires": out[h]["relation_FIRES_on_recovered"],
             "frac": out[h]["frac_recovered_relation_fires"]} for h in out}))

    verdict = {}
    for M in Msweep:
        tot_rec = sum(res[f"M{M}"][h]["n_recovered_golds"] for h in ("1", "2", "3"))
        tot_fire = sum(res[f"M{M}"][h]["relation_FIRES_on_recovered"] for h in ("1", "2", "3"))
        verdict[f"M{M}"] = {"recovered_golds": tot_rec, "relation_fires": tot_fire,
                            "frac": round(tot_fire / max(tot_rec, 1), 4),
                            "CURRENT_EXISTING_EXPERT_SET_LACKS_RECOVERY_SIGNAL":
                                "YES" if (tot_fire / max(tot_rec, 1)) < 0.02 else "NO(needs cosine/rank pass)"}
    final = {"phase": "Q2.1 — exact RELATION expert coverage on recovered multi-hop golds (MetaQA VAL); CPU; no Qwen; TEST untouched",
             "note": "relation FIRES iff title(gold) in query_edges (title-mention connecting sentence over topic-entity content). "
                     "mask=0 => RRF contribution 0 => cannot admit. Coverage answers Q2.1 without the cosine unless fire-rate is non-trivial.",
             "M_values": Msweep, "subset_per_hop": subset, "PER_HOP": res, "VERDICT": verdict,
             "NEW_ENCODER_FORWARD_PASSES": 0, "TARGET_TEST_TOUCHED": "NO"}
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(final, open(outp, "w"), indent=1, default=str)
    log(f"Q2.1_DONE -> {outp}")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
