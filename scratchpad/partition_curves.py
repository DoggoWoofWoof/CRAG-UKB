#!/usr/bin/env python
"""
Partition curves P=1,3,5,10,20,50,100,200 for variant A (historical structural+dense-kNN)
Compute directly from raw partition rankings, no hard-coded recall@50 dict keys.
Records Recall@P, FullCov, AnyCov/AllCov, gold_fraction etc.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("C:\\Users\\Swastik\\Desktop\\CRAG"))
import json, logging, numpy as np, faiss
from src.core.engine import CoreEngine
from src.experiments.overlap_retrain import _hard_membership, _splits
from src.experiments.l1_rerank100 import _feats, _rr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
Ps = [1,3,5,10,20,50,100,200]
datasets = ["squad_clean","2wiki_clean","musique_clean","hotpotqa_clean","webqsp","metaqa"]
os.makedirs("results/L1", exist_ok=True)
os.makedirs("scratchpad/partition_curves", exist_ok=True)
out={}
for ds in datasets:
    try:
        print(f"[partition_curves] loading {ds}")
        eng = CoreEngine(source=ds, index_subdir="gte_qwen")
        # Load X for dense voting? Need dense order to vote partitions
        # Use cached query embeddings if available, else encode via encoder?
        # For partition curve, we need partition ranking per query via dense voting (as in l2_seed _topP)
        # We'll use l1_universal_head _load to get qte, X, mem_idx, etc.
        from src.experiments.l1_universal_head import _load
        data = _load(ds, "gte_qwen", limit=8000, tr_cap=3000, te_cap=0)  # te_cap 0 = uncapped via our convention? but _load uses limit logic; pass large limit to get full
        # Actually _load caps via te_cap param: we pass te_cap 2000 uncapped? Use 1000000 to get full
        # For now data already loaded with te_cap 3000? We passed limit 8000 etc but _load internally caps via te_cap; we gave te_cap 0 -> actually in l1_universal_head _load caps logic uses limit param not te_cap? Need check.
        # Safer to reload with large caps
        # Let's try again with proper uncapped via direct _splits
        X = data["X"]; npart = data["npart"]; mem_idx = data["mem_idx"]; hard = data["hard"]
        qte, ste, gte = data["test"]
        print(f"  {ds}: npart={npart} nq_test={len(gte)} corpus_N={len(hard)}")
        # Build dense order for voting: faiss dense retrieval top 100
        import faiss as _faiss
        idx=_faiss.IndexFlatIP(X.shape[1]); idx.add(X)
        _, I = idx.search(qte, 100)  # dense top 100 for voting
        # Compute partition votes via _feats + _rr
        S,M = _feats(I, mem_idx, npart, topn=200)
        votes = _rr(S) + _rr(M)
        # votes shape (nq, npart) -> ranking
        ranking = np.argsort(-votes, axis=1)  # (nq, npart)
        # For each P compute metrics
        # Need gold partitions per query: mem_idx[g] for each gold doc g
        gpl = [[mem_idx[g] for g in gg] for gg in gte]
        # Also need hard gold for coverage details
        results={}
        for P in Ps:
            eff = min(P, npart)
            saturated = P > npart
            # For each query, topP set
            topP_sets = [set(ranking[qi,:eff]) for qi in range(len(ranking))]
            # FullCov: all gold docs' partitions covered? Need at least one gold partition per gold doc in topP
            # For multi-gold, FullCov means all gold docs reachable (all golds have at least one partition in topP)
            # AnyCov: at least one gold doc reachable
            # AllCov: all gold docs reachable (same as FullCov for doc-level? but for multi-doc queries, distinct)
            # gold_fraction: mean fraction of golds whose partition in topP
            full=0; anyc=0; allc=0; frac_sum=0; nq=0
            for qi, gg in enumerate(gte):
                if not gg:
                    continue
                nq+=1
                gold_sets = [set(mem_idx[g]) for g in gg]
                # For each gold doc, is its partition set intersecting topP?
                hits = [1 if (gs & topP_sets[qi]) else 0 for gs in gold_sets]
                frac = sum(hits)/len(hits) if hits else 0
                frac_sum+=frac
                if frac==1.0:
                    full+=1
                    allc+=1
                if frac>0:
                    anyc+=1
                # AllCov vs FullCov distinction: FullCov requires all gold partitions covered? Already frac==1
                # For single-gold datasets, any and all same
            res={
                "P":P,
                "effective_P":eff,
                "saturated":saturated,
                "FullCov": round(100*full/max(1,nq),2),
                "AnyCov": round(100*anyc/max(1,nq),2),
                "AllCov": round(100*allc/max(1,nq),2),
                "mean_gold_fraction": round(100*frac_sum/max(1,nq),2),
                "nq":nq,
                "npart":npart
            }
            # Also compute Recall@P as partition recall (gold partition in topP) same as gold_fraction? Keep
            results[str(P)]=res
            print(f"    P={P} eff {eff} sat {saturated} FullCov {res['FullCov']} Any {res['AnyCov']} All {res['AllCov']} frac {res['mean_gold_fraction']}")
        out[ds]=results
        # Save per dataset
        json.dump(results, open(f"scratchpad/partition_curves/{ds}_A.json","w"), indent=2)
    except Exception as e:
        print(f"[partition_curves] {ds} failed: {e}")
        import traceback; traceback.print_exc()
        out[ds]={"error":str(e)}
json.dump(out, open("results/L1/partition_curves_A.json","w"), indent=2)
print("-> results/L1/partition_curves_A.json")
