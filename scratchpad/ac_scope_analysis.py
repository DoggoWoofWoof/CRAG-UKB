"""A-vs-C L1 scope/gold overlap + P20-vs-P50 analysis (Tasks 2-4).
CPU-only, reuses cached dense/splade top200 + partition maps. No retrieval recompute.
Routing is bit-identical to scratchpad/l1_eval_phase1.py (fusion RRF K0=60, mem_idx = structural one-hop)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from l1_eval_phase1 import load_partition_topology  # verified routing/topology loader

K0 = 60

def _partition_ranking(order_K, mem_idx, npart, K):
    nq = len(order_K)
    S = np.zeros((nq, npart), dtype=np.float32)
    M = np.zeros((nq, npart), dtype=np.float32)
    for qi in range(nq):
        for r, nd in enumerate(order_K[qi][:K]):
            w = 1.0/(K0 + r)
            if nd < 0 or nd >= len(mem_idx):
                continue
            for p in mem_idx[int(nd)]:
                S[qi, p] += w
                if w > M[qi, p]:
                    M[qi, p] = w
    def rr(score):
        order = np.argsort(-score, axis=1)
        rank = np.empty((nq, npart), dtype=np.int32)
        rows = np.arange(nq)[:, None]
        rank[rows, order] = np.arange(npart)[None, :]
        return 1.0/(K0 + rank)
    votes = rr(S) + rr(M)
    return np.argsort(-votes, axis=1).astype(np.int32)

def _fuse(dense_ranking, splade_ranking, npart):
    nq = dense_ranking.shape[0]; rows = np.arange(nq)[:, None]; cols = np.arange(npart)[None, :]
    pos_d = np.empty((nq, npart), dtype=np.int32); pos_d[rows, dense_ranking] = cols
    pos_s = np.empty((nq, npart), dtype=np.int32); pos_s[rows, splade_ranking] = cols
    fv = 1.0/(K0 + pos_d) + 1.0/(K0 + pos_s)
    fv_d = np.take_along_axis(fv, dense_ranking, axis=1)
    idx2 = np.argsort(-fv_d, axis=1, kind="stable")
    return np.take_along_axis(dense_ranking, idx2, axis=1).astype(np.int32)

def _fusion_ranking(dense_all, splade_all, mem_idx, npart, K):
    dr = _partition_ranking(dense_all[:, :K], mem_idx, npart, K)
    sr = _partition_ranking(splade_all[:, :K], mem_idx, npart, K)
    return _fuse(dr, sr, npart)

def _test_idx(dataset, j):
    si = j["split_indices"]
    if dataset == "webqsp":
        return si["all"]
    return si["test"] if si.get("test") else si["all"]

def analyze(dataset, K=100):
    base = f"data/ukb_storage/{dataset}/gte_qwen"
    j = json.load(open(f"{base}/query_ids_all.json"))
    golds = j["golds"]; hops = j.get("hops", [None]*len(j["ids"]))
    test_idx = _test_idx(dataset, j)
    dense = np.load(f"{base}/dense_top200_all.npy", mmap_mode='r')
    splade = np.load(f"{base}/splade_top200_all.npy", mmap_mode='r')
    dense_t = np.array([dense[i] for i in test_idx])
    splade_t = np.array([splade[i] for i in test_idx])

    hA, memA, npA, doc_nodes, _ = load_partition_topology(dataset, "A")
    hC, memC, npC, _, _ = load_partition_topology(dataset, "C")
    id2idx = {n.node_id: i for i, n in enumerate(doc_nodes)}
    # gold doc indices + hops for test queries
    tg, th = [], []
    for i in test_idx:
        gi = [id2idx[g] for g in golds[i] if g in id2idx]
        tg.append(gi); th.append(hops[i])

    rankA = _fusion_ranking(dense_t, splade_t, memA, npA, K)
    rankC = _fusion_ranking(dense_t, splade_t, memC, npC, K)

    # contingency M[pa,pc] = #docs with hard_A=pa AND hard_C=pc  (exact intersection helper)
    from scipy.sparse import csr_matrix
    N = len(hA)
    M = csr_matrix((np.ones(N, np.int64), (hA, hC)), shape=(npA, npC)).toarray()
    sizeA = np.bincount(hA, minlength=npA)
    sizeC = np.bincount(hC, minlength=npC)

    res = {"dataset": dataset, "n_test": len(test_idx), "K": K}
    for P in (20, 50):
        jacc = []; ident = 0; aonly_frac = []; conly_frac = []
        szA_l = []; szC_l = []; inter_l = []; uni_l = []
        # gold rescue (ANY + ALL)
        both=aA=cC=neither=0          # ANY
        both_all=aA_all=cC_all=neither_all=0   # ALL
        gfA_sum=gfC_sum=0.0
        for qi in range(len(test_idx)):
            selA = rankA[qi][:P]; selC = rankC[qi][:P]
            szA = int(sizeA[selA].sum()); szC = int(sizeC[selC].sum())
            inter = int(M[np.ix_(selA, selC)].sum())
            uni = szA + szC - inter
            szA_l.append(szA); szC_l.append(szC); inter_l.append(inter); uni_l.append(uni)
            jacc.append(inter/uni if uni else 1.0)
            if szA == szC == inter: ident += 1
            aonly_frac.append((szA-inter)/uni if uni else 0.0)
            conly_frac.append((szC-inter)/uni if uni else 0.0)
            gi = tg[qi]
            if not gi:  # no mapped gold -> skip from rescue accounting
                continue
            setA = set(int(x) for x in selA); setC = set(int(x) for x in selC)
            ga = [int(hA[g]) in setA for g in gi]
            gc = [int(hC[g]) in setC for g in gi]
            anyA, anyC = any(ga), any(gc)
            allA, allC = all(ga), all(gc)
            gfA_sum += sum(ga)/len(gi); gfC_sum += sum(gc)/len(gi)
            if anyA and anyC: both+=1
            elif anyA: aA+=1
            elif anyC: cC+=1
            else: neither+=1
            if allA and allC: both_all+=1
            elif allA: aA_all+=1
            elif allC: cC_all+=1
            else: neither_all+=1
        ng = both+aA+cC+neither
        res[f"P{P}"] = {
            "mean_jaccard": round(float(np.mean(jacc)),4),
            "median_jaccard": round(float(np.median(jacc)),4),
            "identical_scope_pct": round(100*ident/len(test_idx),2),
            "A_only_frac_mean": round(100*float(np.mean(aonly_frac)),2),
            "C_only_frac_mean": round(100*float(np.mean(conly_frac)),2),
            "mean_scopeA": round(float(np.mean(szA_l)),1),
            "mean_scopeC": round(float(np.mean(szC_l)),1),
            "mean_inter": round(float(np.mean(inter_l)),1),
            "mean_union": round(float(np.mean(uni_l)),1),
            "gold_rescue_ANY": {"both_success":both,"A_only":aA,"C_only":cC,"both_fail":neither,"n_gold_q":ng},
            "gold_rescue_ALL": {"both_success":both_all,"A_only":aA_all,"C_only":cC_all,"both_fail":neither_all},
            "ANY_A_pct": round(100*(both+aA)/ng,2), "ANY_C_pct": round(100*(both+cC)/ng,2),
            "ALL_A_pct": round(100*(both_all+aA_all)/ng,2), "ALL_C_pct": round(100*(both_all+cC_all)/ng,2),
            "GoldFrac_A": round(100*gfA_sum/ng,2), "GoldFrac_C": round(100*gfC_sum/ng,2),
        }
    # P20 vs P50 within each topology (A and C separately)
    for topo, rank, hard, size, npart in (("A",rankA,hA,sizeA,npA),("C",rankC,hC,sizeC,npC)):
        pv = {}
        for P in (20,50):
            anyc=allc=0; scopes=[]
            for qi in range(len(test_idx)):
                sel=set(int(x) for x in rank[qi][:P])
                scopes.append(int(size[list(sel)].sum()))
                gi=tg[qi]
                if not gi: continue
                r=[int(hard[g]) in sel for g in gi]
                if any(r): anyc+=1
                if all(r): allc+=1
            pv[P]={"ANY":round(100*anyc/len(test_idx),2),"ALL":round(100*allc/len(test_idx),2),
                   "mean_scope":round(float(np.mean(scopes)),1)}
        d_any=pv[50]["ANY"]-pv[20]["ANY"]; d_all=pv[50]["ALL"]-pv[20]["ALL"]
        dscope=pv[50]["mean_scope"]-pv[20]["mean_scope"]
        # primary metric per dataset
        prim = "ALL" if dataset in ("2wiki_clean","musique_clean","hotpotqa_clean") else "ANY"
        dprim = pv[50][prim]-pv[20][prim]
        res[f"P20vP50_{topo}"] = {
            "P20":pv[20],"P50":pv[50],"primary":prim,
            "delta_primary_pp":round(dprim,2),
            "scope_multiplier":round(pv[50]["mean_scope"]/max(pv[20]["mean_scope"],1),2),
            "cov_gain_per_1k_extra":round(1000*dprim/max(dscope,1),3),
        }
    return res

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["2wiki_clean","musique_clean","hotpotqa_clean","squad_clean","metaqa","webqsp"])
    ap.add_argument("--out", default="results/L1/ac_scope_analysis.json")
    a = ap.parse_args()
    allres = {}
    for ds in a.datasets:
        print(f"=== {ds} ===", flush=True)
        r = analyze(ds)
        allres[ds] = r
        os.makedirs("results/L1", exist_ok=True)
        json.dump(allres, open(a.out, "w"), indent=2)
        p50 = r["P50"]
        print(f"  Jaccard mean {p50['mean_jaccard']} identical {p50['identical_scope_pct']}% "
              f"| ANY A/C {p50['ANY_A_pct']}/{p50['ANY_C_pct']} "
              f"rescue A_only {p50['gold_rescue_ANY']['A_only']} C_only {p50['gold_rescue_ANY']['C_only']}", flush=True)
    print(f"\nWROTE {a.out}")
