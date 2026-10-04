"""STEP 5 -- PRIVILEGED_METAQA_DIAGNOSTIC.  MetaQA ONLY.  NOT a valid universal method.

Purpose: decompose the old post-P50 headline (hop2 0.708 -> 0.928 = +22pt, hop3 0.288 -> 0.424 = +14pt,
results/GENERALIZATION/_g2_smoke_s1c.json) into its three ingredients:

    (i)   PRIVILEGED SEEDING     bracketed `[entity]` in the question text -> metaqa_ent_<name> row
                                 (a benchmark annotation; forbidden in the universal mainline)
    (ii)  PRIVILEGED ADJACENCY   the official MetaQA KB edge list data/original/metaqa/kb.txt
                                 (the mainline uses master_nodes.neighbors uniformly, no dataset branch)
    (iii) ADDITION BUDGET        the old run added ~316-436 nodes/query (B_CAP=2000, beam 256), not M

PART 1  exact reproduction of the published numbers on the published sample (old code path, verbatim).
PART 2  controlled 2 x 2 x budget decomposition on the MAIN dev sample with ONE implementation
        (TA.expand_dir over CSR adjacency) so that seeds and adjacency are the only things that vary.

No gold anywhere in seeding/expansion/selection. TEST never touched.

  python scratchpad/_ta_priv_metaqa.py
"""
import os, sys, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA

DS = "metaqa"
EMB = f"data/ukb_storage/{DS}/gte_qwen/"
OUT = "results/GENERALIZATION/_g2_priv_metaqa.json"
M_BUDGETS = [8, 16, 32, 64]
OLD_BEAM = 256          # the beam the published headline used
OLD_BCAP = 2000         # the published run's total-candidate cap
EVAL_CAP = int(os.environ.get("TA_CAP", "2000"))
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
_BRK = re.compile(r"\[(.+?)\]")


# ----------------------------------------------------------------- old code path (verbatim, g2_smoke_s1c)
def old_residual(qvec, seed_rows, Xn):
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    if not seed_rows:
        return q
    E = Xn[seed_rows]
    U, s, _ = np.linalg.svd(E.T, full_matrices=False); U = U[:, s > 1e-6]
    if U.shape[1] == 0:
        return q
    r = q - U @ (U.T @ q); n = np.linalg.norm(r)
    return r / n if n > 1e-6 else q


def old_expand_dir(seed_rows, r_q, adj, M, Xn, H_BUDGET=3, B=OLD_BCAP):
    scope = set(seed_rows); frontier = list(seed_rows); scored = []; escored = 0
    for hop in range(H_BUDGET):
        cand = []; seen = set()
        for e in frontier:
            xe = Xn[e]
            for (v, rel, dirn) in adj.get(e, []):
                escored += 1
                if v in scope or (e, v) in seen:
                    continue
                seen.add((e, v))
                delta = Xn[v] - xe; nd = np.linalg.norm(delta)
                if nd < 1e-9:
                    continue
                cand.append((float(r_q @ (delta / nd)), v))
        if not cand:
            break
        cand.sort(key=lambda x: -x[0]); keep = cand[:M]
        nf = []
        for sdir, v in keep:
            if v not in scope:
                scope.add(v); nf.append(v); scored.append((sdir, v))
        frontier = nf
        if len(scope) >= B * 2:
            break
    out = []
    for sdir, v in sorted(scored, key=lambda x: -x[0]):
        if len(out) + len(seed_rows) >= B:
            break
        out.append(v)
    return out, escored


def build_kb_adj(id2row):
    """official MetaQA KB edge list in nodes.npy row space (PRIVILEGED -- dataset-specific)."""
    adj = {}; n = 0
    norm = lambda s: "metaqa_ent_" + s.strip().lower()
    for line in open("data/original/metaqa/kb.txt", encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) != 3:
            continue
        h, rel, t = p
        hi = id2row.get(norm(h)); ti = id2row.get(norm(t))
        if hi is None or ti is None:
            continue
        adj.setdefault(hi, []).append((ti, rel, +1))
        adj.setdefault(ti, []).append((hi, rel, -1))
        n += 1
    return adj, n


def kb_adj_to_csr(adj, ndocs):
    """same edges, CSR form, so PART 2 can hold the traversal implementation fixed."""
    sets = [set() for _ in range(ndocs)]
    for i, lst in adj.items():
        for (v, _r, _d) in lst:
            if v != i:
                sets[i].add(v); sets[v].add(i)
    deg = np.array([len(s) for s in sets], np.int32)
    ptr = np.zeros(ndocs + 1, np.int64); ptr[1:] = np.cumsum(deg)
    idx = np.empty(int(ptr[-1]), np.int32)
    for i, s in enumerate(sets):
        if s:
            idx[ptr[i]:ptr[i + 1]] = np.fromiter(sorted(s), np.int32, len(s))
    return ptr, idx, deg


def main():
    hard, mem, npart, (adjp, adji), deg, id2row = TA.load_topology(DS, log)
    part_sizes = np.bincount(hard[hard >= 0], minlength=npart)
    ndocs = len(hard)
    j = json.load(open(EMB + "query_ids_all.json"))
    ids, golds, hops, si = j["ids"], j["golds"], j["hops"], j["split_indices"]

    from src.pipeline.standardizer import load_nodes
    alln = load_nodes(f"data/processed/master_nodes_{DS}.json")
    qtext = {n.node_id: n.content for n in alln if n.metadata.get("type") == "question"}
    del alln
    kb_adj, n_kb = build_kb_adj(id2row)
    log(f"kb.txt edges mapped into row space: {n_kb}  nodes_with_edges={len(kb_adj)}")

    nodes = np.load(EMB + "nodes.npy", mmap_mode="r")
    qall = np.load(EMB + "queries_all.npy", mmap_mode="r")
    dense_all = np.load(EMB + "dense_top200_all.npy", mmap_mode="r")
    splade_all = np.load(EMB + "splade_top200_all.npy", mmap_mode="r")
    Xn = np.array(nodes, np.float32)
    for a in range(0, len(Xn), 100000):
        b = min(a + 100000, len(Xn)); Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)

    def priv_seeds(row):
        out = []
        for m in _BRK.findall(qtext[ids[row]]):
            r = id2row.get("metaqa_ent_" + m.strip().lower())
            if r is not None:
                out.append(r)
        return out

    def base_of(rws):
        dK = np.stack([np.asarray(dense_all[i][:TA.K_LOCK]) for i in rws]).astype(np.int64)
        sK = np.stack([np.asarray(splade_all[i][:TA.K_LOCK]) for i in rws]).astype(np.int64)
        rk = TA.rrf_partitions([TA.partition_ranking(list(dK), mem, npart),
                                TA.partition_ranking(list(sK), mem, npart)], npart)
        return dK, sK, [set(int(x) for x in rk[q][:TA.P_MAIN]) for q in range(len(rws))]

    RES = {}

    # ================================================================= PART 1 : exact reproduction
    log("PART 1: exact reproduction of the published run (sample = first 250 val queries per hop) ...")
    samp = {h: [] for h in (1, 2, 3)}
    for i in si["val"]:
        h = hops[i]
        if h in samp and len(samp[h]) < 250:
            samp[h].append(i)
    repro = {}
    for h in (1, 2, 3):
        rws = samp[h]
        _dK, _sK, bsel = base_of(rws)
        p50 = uni = 0; nfe = 0; addc = []
        for qi, i in enumerate(rws):
            g = [id2row[x] for x in golds[i] if x in id2row]
            if not g:
                continue
            nfe += 1
            sd = priv_seeds(i)
            r_q = old_residual(np.asarray(qall[i]), sd, Xn)
            exp, _e = old_expand_dir(sd, r_q, kb_adj, OLD_BEAM, Xn)
            E = set(exp) | set(sd)
            addc.append(len(E) - len(set(sd)))
            inp = [int(hard[x]) in bsel[qi] for x in g]
            p50 += int(all(inp)); uni += int(all(a or (x in E) for a, x in zip(inp, g)))
        repro[f"hop{h}"] = {"n": nfe, "P50_ALL": round(p50 / nfe, 4), "UNION_ALL": round(uni / nfe, 4),
                            "added_mean": round(float(np.mean(addc)), 1)}
        log(f"  hop{h}: P50 {repro[f'hop{h}']['P50_ALL']:.4f} -> UNION {repro[f'hop{h}']['UNION_ALL']:.4f} "
            f"(+{repro[f'hop{h}']['added_mean']:.0f} nodes)")
    pub = json.load(open("results/GENERALIZATION/_g2_smoke_s1c.json"))["RESULTS"]
    repro["PUBLISHED"] = {f"hop{h}": {"P50_ALL": pub[f"M256|h{h}"]["P50_ALL"],
                                      "UNION_ALL": pub[f"M256|h{h}"]["UNION_ALL"],
                                      "added_mean": pub[f"M256|h{h}"]["added_cand_mean"]} for h in (1, 2, 3)}
    repro["MATCHES_PUBLISHED"] = all(
        abs(repro[f"hop{h}"]["UNION_ALL"] - pub[f"M256|h{h}"]["UNION_ALL"]) < 5e-3 for h in (1, 2, 3))
    RES["PART1_exact_reproduction"] = repro

    # ================================================================= PART 2 : controlled decomposition
    np.random.seed(0)
    rows = si["val"]
    buck = {1: [], 2: [], 3: []}
    for r in rows:
        if hops[r] in buck:
            buck[hops[r]].append(r)
    per = EVAL_CAP // 3; sel = []
    for h in (1, 2, 3):
        b = list(buck[h]); np.random.shuffle(b); sel += b[:per]
    rows = sorted(sel)
    keep, gold_rows = [], []
    for r in rows:
        g = [id2row[x] for x in golds[r] if x in id2row]
        if g:
            keep.append(r); gold_rows.append(g)
    rows = keep; nq = len(rows)
    dK, sK, base_sel = base_of(rows)
    d200 = np.stack([np.asarray(dense_all[i]) for i in rows]).astype(np.int64)
    s200 = np.stack([np.asarray(splade_all[i]) for i in rows]).astype(np.int64)
    fused = [TA.node_rrf(d200[q], s200[q]) for q in range(nq)]
    base_scope = np.array([int(part_sizes[sorted(s)].sum()) for s in base_sel], np.int64)
    ind_base = np.zeros(nq, np.int8)
    for q in range(nq):
        ind_base[q] = int(all(int(hard[g]) in base_sel[q] for g in gold_rows[q]))
    log(f"PART 2: main dev sample n={nq} BASE_ALL={ind_base.mean():.4f}")

    kbp, kbi, kbd = kb_adj_to_csr(kb_adj, ndocs)
    ADJ = {"KB_privileged": (kbp, kbi, kbd), "MASTER_universal": (adjp, adji, deg)}
    SEED = {"PRIVILEGED_bracket": lambda q: priv_seeds(rows[q]),
            "UNIVERSAL_RRF": lambda q: fused[q][:TA.SEED_K],
            "UNIVERSAL_DENSE": lambda q: [int(x) for x in dK[q][:TA.SEED_K]],
            "UNIVERSAL_SPLADE": lambda q: [int(x) for x in sK[q][:TA.SEED_K]]}

    def run(seed_name, adj_name, beam):
        ap, ai, ad = ADJ[adj_name]; sf = SEED[seed_name]
        out = []; esc = 0; t = time.time()
        for q in range(nq):
            sd = sf(q)
            if not sd:
                out.append([]); continue
            r_q = TA.residual(np.asarray(qall[rows[q]], np.float64), sd, Xn)
            a, _vm, e = TA.expand_dir(sd, r_q, ap, ai, ad, Xn, beam)
            esc += e
            out.append([v for v in a if int(hard[v]) not in base_sel[q]])
        return out, {"edges_traversed": int(esc), "sec": round(time.time() - t, 1),
                     "residual_available_per_query": round(sum(len(x) for x in out) / nq, 1)}

    def sc(added, tag):
        allc = 0; hop_all = {}; hop_n = {}; ind = np.zeros(nq, np.int8); na = []
        newly = 0; grec = 0
        for q in range(nq):
            A = set(added[q]); na.append(len(A))
            cov = [(int(hard[g]) in base_sel[q]) or (g in A) for g in gold_rows[q]]
            bc = [(int(hard[g]) in base_sel[q]) for g in gold_rows[q]]
            grec += sum(1 for a_, b_ in zip(cov, bc) if a_ and not b_)
            ok = all(cov); allc += ok; ind[q] = int(ok)
            newly += int(ok and not ind_base[q])
            k = str(hops[rows[q]]); hop_n[k] = hop_n.get(k, 0) + 1
            hop_all[k] = hop_all.get(k, 0) + int(ok)
        from math import comb
        w = int(((ind == 1) & (ind_base == 0)).sum()); l = int(((ind == 0) & (ind_base == 1)).sum())
        n2 = w + l
        p = 1.0 if n2 == 0 else min(1.0, 2 * sum(comb(n2, i) for i in range(min(w, l) + 1)) / 2.0 ** n2)
        return {"tag": tag, "ALL": round(allc / nq, 4), "dALL_vs_BASE": round(allc / nq - ind_base.mean(), 4),
                "queries_newly_ALL_covered": int(newly), "newly_recovered_gold_occurrences": int(grec),
                "added_nodes_mean": round(float(np.mean(na)), 1),
                "final_scope_nodes_mean": round(float((base_scope + np.array(na)).mean()), 1),
                "per_hop_ALL": {h: round(hop_all[h] / hop_n[h], 4) for h in sorted(hop_n)},
                "net": w - l, "mcnemar_p": round(p, 5)}

    D = {}; COST = {}
    D["BASE"] = sc([[] for _ in range(nq)], "BASE")
    for adj_name in ADJ:
        for seed_name in SEED:
            if seed_name.startswith("UNIVERSAL_") and seed_name != "UNIVERSAL_RRF" \
                    and adj_name == "KB_privileged":
                continue                       # dense/splade seed ablations only on the mainline adjacency
            res, c = run(seed_name, adj_name, OLD_BEAM)
            COST[f"{seed_name}|{adj_name}|beam{OLD_BEAM}"] = c
            for M in M_BUDGETS:
                D[f"{seed_name}|{adj_name}|M{M}"] = sc([x[:M] for x in res], f"{seed_name}|{adj_name}|M{M}")
            D[f"{seed_name}|{adj_name}|UNBOUNDED"] = sc(res, f"{seed_name}|{adj_name}|UNBOUNDED")
            log(f"  {seed_name:20s} {adj_name:18s} avail/q={c['residual_available_per_query']:6.1f} "
                f"M32 dALL={D[f'{seed_name}|{adj_name}|M32']['dALL_vs_BASE']:+.4f}  "
                f"UNB dALL={D[f'{seed_name}|{adj_name}|UNBOUNDED']['dALL_vs_BASE']:+.4f}")
    RES["PART2_controlled_decomposition"] = D
    RES["PART2_COST"] = COST
    RES["_meta"] = {"n_dev_queries": nq, "BASE_ALL": round(float(ind_base.mean()), 4),
                    "BASE_SCOPE_NODES": round(float(base_scope.mean()), 1),
                    "beam": OLD_BEAM, "M_BUDGETS": M_BUDGETS,
                    "kb_edges_mapped": n_kb,
                    "WARNING": "PRIVILEGED_* rows use benchmark entity annotations and/or the official "
                               "MetaQA KB file. They are a DIAGNOSTIC. Not a universal method. Not promotable."}
    json.dump(RES, open(OUT, "w"), indent=1)

    print(f"\n=== PRIVILEGED METAQA DIAGNOSTIC  n={nq}  BASE_ALL={ind_base.mean():.4f} ===")
    print("PART 1 exact reproduction of published headline: "
          f"MATCHES_PUBLISHED={repro['MATCHES_PUBLISHED']}")
    for h in (1, 2, 3):
        r = repro[f"hop{h}"]; p = repro["PUBLISHED"][f"hop{h}"]
        print(f"   hop{h}  mine P50 {r['P50_ALL']:.4f} -> UNION {r['UNION_ALL']:.4f} (+{r['added_mean']:.0f})"
              f"   |  published {p['P50_ALL']:.4f} -> {p['UNION_ALL']:.4f} (+{p['added_mean']:.0f})")
    print("\nPART 2 controlled decomposition (same traversal implementation, main dev sample)")
    print(f"  {'seeds | adjacency':44s} {'M8':>9s} {'M16':>9s} {'M32':>9s} {'M64':>9s} {'UNBOUND':>9s} "
          f"{'hop2':>7s} {'hop3':>7s} {'added':>7s}")
    print(f"  {'BASE':44s} " + " ".join(f"{D['BASE']['ALL']:9.4f}" for _ in range(5)) +
          f" {D['BASE']['per_hop_ALL']['2']:7.4f} {D['BASE']['per_hop_ALL']['3']:7.4f} {0:7.0f}")
    for k in [k for k in D if k != "BASE" and k.endswith("UNBOUNDED")]:
        pre = k.rsplit("|", 1)[0]
        cells = " ".join(f"{D[f'{pre}|M{M}']['ALL']:9.4f}" for M in M_BUDGETS)
        u = D[k]
        print(f"  {pre:44s} {cells} {u['ALL']:9.4f} {u['per_hop_ALL']['2']:7.4f} "
              f"{u['per_hop_ALL']['3']:7.4f} {u['added_nodes_mean']:7.1f}")
    print("wrote " + OUT)


if __name__ == "__main__":
    main()
