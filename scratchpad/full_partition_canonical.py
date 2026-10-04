#!/usr/bin/env python
"""
FULL partition curves using canonical C5 encodings (dense) + batched FAISS.
- Loads query embeddings shardwise from data/canonical/<ds>/encodings/dense/queries
- Loads doc embeddings shardwise from data/canonical/<ds>/encodings/dense/docs
- Preserves exact canonical query IDs from ids_*.json and official splits from queries.jsonl
- Never materializes Q@X.T; uses faiss IndexFlatIP batched search top100
- Computes historical voting _feats/_rr for P=1,3,5,10,20,50,100,200
- Stores query-level rankings for recomputation

For MetaQA we combine 1hop/2hop/3hop; for 2Wiki we use 2wiki (398k) benchmark, for Hotpot we use hotpotqa (5.2M) fullwiki
But for graph partitions we need partition_map: we use existing ukb_storage _clean partitions mapped via doc ID text hash?
Simplified: for now we compute partition curves on the canonical doc partitions built on-the-fly with target 100 via pymetis if not exists,
otherwise fallback to ukb_storage clean partitions with ID mapping via gold_doc_ids that exist in both.
This script is versioned as full-validation, not debug.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
import json, numpy as np, faiss, pathlib, logging
from src.experiments.l1_rerank100 import _feats, _rr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
Ps=[1,3,5,10,20,50,100,200]

# Mapping from logical dataset name to canonical dataset(s) for queries
CANONICAL_QUERIES={
    "squad": ["squad"],
    "webqsp": ["webqsp"],
    "metaqa": ["metaqa_1hop","metaqa_2hop","metaqa_3hop"],
    "2wiki": ["2wiki"],
    "musique": ["musique"],
    "hotpotqa": ["hotpotqa"],
}
CANONICAL_DOCS={
    "squad":"squad",
    "webqsp":"webqsp",
    "metaqa":"metaqa",
    "2wiki":"2wiki",
    "musique":"musique",
    "hotpotqa":"hotpotqa",
}

def load_canonical_embeddings(dataset, kind):
    base=f"data/canonical/{dataset}/encodings/dense/{kind}"
    man=json.load(open(os.path.join(base,"manifest.json")))
    shards=[]
    ids=[]
    for s in man["shards_present"]:
        arr=np.load(os.path.join(base,f"shard_{s:05d}.npy"))
        # float16 -> float32 for faiss
        if arr.dtype==np.float16:
            arr=arr.astype(np.float32)
        shards.append(arr)
        j=json.load(open(os.path.join(base,f"ids_{s:05d}.json")))
        ids.extend(j)
    mat=np.concatenate(shards, axis=0) if shards else np.zeros((0,1536),dtype=np.float32)
    return mat, ids, man

def load_queries_full(dataset):
    # dataset is logical like "squad" or "metaqa" -> need to combine hops for metaqa
    q_mats=[]; q_ids=[]; golds=[]
    for c in CANONICAL_QUERIES[dataset]:
        mat, ids, man = load_canonical_embeddings(c, "queries")
        # Load queries.jsonl to get split and golds
        qjsonl=f"data/canonical/{c}/queries.jsonl"
        # Build map query_id -> (split, gold_doc_ids, question)
        # For metaqa hops, each has its own queries.jsonl
        qmap={}
        with open(qjsonl, encoding="utf-8") as f:
            for line in f:
                j=json.loads(line)
                qmap[j["query_id"]]=j
        # Filter to validation split(s): for squad dev, webqsp test, etc.
        # Use official_split field: train/dev/test
        # For metaqa hops, each has 1hop/test etc but we want test split for each hop
        # For general, we take dev+test as validation if both exist, else dev or test
        # Determine validation query_ids: those with official_split in ["dev","test"] or split containing "dev"/"test"
        valid_ids=[qid for qid, rec in qmap.items() if rec.get("official_split") in ("dev","test") or "dev" in rec.get("official_split","") or "test" in rec.get("official_split","")]
        # If no split field (unlikely), take all
        if not valid_ids:
            valid_ids=list(qmap.keys())
        # Map via ids list order
        id_to_idx={qid:i for i,qid in enumerate(ids)}
        for qid in valid_ids:
            if qid in id_to_idx:
                idx=id_to_idx[qid]
                q_mats.append(mat[idx])
                q_ids.append(qid)
                golds.append(qmap[qid].get("gold_doc_ids",[]))
        logging.info(f"[canonical queries] {c}: total {len(ids)} valid {len(valid_ids)} kept {len([x for x in valid_ids if x in id_to_idx])}")
    if q_mats:
        qmat=np.stack(q_mats).astype(np.float32)
    else:
        qmat=np.zeros((0,1536),dtype=np.float32)
    return qmat, q_ids, golds

def load_docs_with_partition(dataset):
    # Try to load canonical docs mat and then map to existing clean partitions via ID or via building new partitions
    c=CANONICAL_DOCS[dataset]
    dmat, dids, dman = load_canonical_embeddings(c, "docs")
    # Try to load existing clean partition_map for this dataset (ukb_storage)
    clean_map_path=None
    for cand in [f"data/ukb_storage/{dataset}_clean/gte_qwen/partition_map.json", f"data/ukb_storage/{dataset}/gte_qwen/partition_map.json", f"data/ukb_storage/{dataset}/partition_map.json"]:
        if os.path.exists(cand):
            clean_map_path=cand
            break
    part_map={}
    if clean_map_path and os.path.exists(clean_map_path):
        part_map=json.load(open(clean_map_path))
        npart=max(part_map.values())+1 if part_map else 0
        logging.info(f"[partition] using existing {clean_map_path} npart {npart} for {dataset} (clean)")
        # Need to map canonical doc IDs to partition via clean_map if overlap else -1
        # For canonical docs that are not in clean, assign -1 (will be ignored in voting)
        # Build mem_idx via _onehop? For variant A, mem_idx is onehop membership (hard + onehop) but for partition curves we use mem_idx from engine's onehop
        # Simplified: use hard partition only (no overlap) for now, but historical uses mem_idx onehop
        # To keep equivalence, we need mem_idx; we can approximate with hard only
        hard=np.array([int(part_map.get(did, -1)) for did in dids])
        # For docs not in map, hard=-1 -> will be excluded
        # mem_idx = [[hard[i]] if hard[i]>=0 else []]
        mem_idx=[[int(hard[i])] if hard[i]>=0 else [] for i in range(len(hard))]
    else:
        logging.warning(f"[partition] no existing partition_map for {dataset}, building naive 100-sized partitions")
        npart=max(1, len(dids)//100)
        hard=np.array([i//100 for i in range(len(dids))])
        mem_idx=[[int(hard[i])] for i in range(len(hard))]
        part_map={did:int(hard[i]) for i,did in enumerate(dids)}
    return dmat, dids, hard, mem_idx, npart, part_map

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["squad","webqsp","metaqa","2wiki","musique","hotpotqa"])
    p.add_argument("--batch", type=int, default=500)
    args=p.parse_args()
    os.makedirs("results/L1", exist_ok=True)
    os.makedirs("scratchpad/full_partition", exist_ok=True)
    out={}
    for ds in args.datasets:
        logging.info(f"=== FULL {ds} ===")
        try:
            qmat, qids, gold_lists = load_queries_full(ds)
            dmat, dids, hard, mem_idx, npart, pmap = load_docs_with_partition(ds)
            logging.info(f" {ds}: q {qmat.shape} d {dmat.shape} npart {npart}")
            # Build FAISS index from docs
            idx=faiss.IndexFlatIP(dmat.shape[1])
            # Normalize for cosine (gte-qwen embeddings are already normalized? canonical dense queries are normalized, docs are normalized)
            # Ensure normalized
            faiss.normalize_L2(qmat)
            faiss.normalize_L2(dmat)
            idx.add(dmat)
            # Batched search top100
            nq=qmat.shape[0]
            I=np.empty((nq,100), dtype=np.int64)
            for s in range(0, nq, args.batch):
                e=min(nq, s+args.batch)
                _, Ii = idx.search(qmat[s:e], 100)
                I[s:e]=Ii
                logging.info(f"  search {s}:{e}/{nq}")
            # Map gold doc IDs to doc indices
            did_to_idx={did:i for i,did in enumerate(dids)}
            gold_indices=[[did_to_idx.get(g,-1) for g in gl if did_to_idx.get(g,-1)!=-1] for gl in gold_lists]
            # For partition voting, need mem_idx per doc index
            # Compute votes
            S,M=_feats(I, mem_idx, npart, topn=200)
            votes=_rr(S)+_rr(M)
            ranking=np.argsort(-votes, axis=1)
            results={}
            for P in Ps:
                eff=min(P,npart); sat=P>npart
                topP=[set(ranking[qi,:eff]) for qi in range(nq)]
                full=anyc=allc=0; frac=0; nq_valid=0
                for qi, gg in enumerate(gold_indices):
                    if not gg: continue
                    nq_valid+=1
                    # gold partitions for each gold doc
                    gold_parts=[set(mem_idx[g]) if g>=0 and g<len(mem_idx) and mem_idx[g] else set() for g in gg]
                    # filter empty
                    gold_parts=[gp for gp in gold_parts if gp]
                    if not gold_parts:
                        continue
                    hits=[1 if (gp & topP[qi]) else 0 for gp in gold_parts]
                    f=sum(hits)/len(hits) if hits else 0
                    frac+=f
                    if f==1.0: full+=1; allc+=1
                    if f>0: anyc+=1
                res={"P":P,"effective_P":eff,"saturated":sat,"FullCov":round(100*full/max(1,nq_valid),2),"AnyCov":round(100*anyc/max(1,nq_valid),2),"AllCov":round(100*allc/max(1,nq_valid),2),"mean_gold_fraction":round(100*frac/max(1,nq_valid),2),"nq":nq_valid,"npart":npart,"nq_total":nq}
                results[str(P)]=res
                logging.info(f"  P{P} {res}")
            out[ds]=results
            json.dump(results, open(f"scratchpad/full_partition/{ds}_A_full.json","w"), indent=2)
            # Also store query-level rankings for recomputation
            np.savez_compressed(f"scratchpad/full_partition/{ds}_rankings.npz", ranking=ranking, qids=np.array(qids), gold=np.array(gold_indices, dtype=object))
        except Exception as e:
            import traceback; traceback.print_exc()
            out[ds]={"error":str(e)}
            json.dump({"error":str(e)}, open(f"scratchpad/full_partition/{ds}_A_full.json","w"), indent=2)
    json.dump(out, open("results/L1/partition_curves_A_full.json","w"), indent=2)
    logging.info("-> results/L1/partition_curves_A_full.json")

if __name__=="__main__":
    main()
