"""Route A step 3 -- union NSM's per-question subgraphs and apply the pre-registered verdict.

    python scratchpad/final_canonical_build/_acquisition/nsm_union.py

Pre-registration: data/final_canonical/webqsp/NSM_ACQUISITION_PREREGISTRATION.json
    EXACT_PREIMAGE     triples == 8,309,195 and relations == 7,058
    SIBLING_EXTRACTION within 10% on BOTH axes
    REFUTED            outside that band
The band was fixed before the measurement and is NOT adjusted here.

Layout (confirmed by inspect_nsm.py):
    <ds>/entities.txt   line i = the MID for local entity id i   (webqsp 1,441,420 / CWQ 2,429,346, >99.98% MIDs)
    <ds>/relations.txt  line i = the relation name for local relation id i (webqsp 6,102 / CWQ 6,649)
    <ds>/{train,dev,test}_simple.json  JSON-lines; each line has subgraph.tuples = [[h_id, rel_id, t_id], ...]

The two datasets have SEPARATE local id spaces, so ids are mapped to strings and re-interned into
one global space before the union -- unioning raw local ids would silently merge unrelated entities.

Codes are packed into int64 numpy arrays rather than a Python set: the raw tuple count is in the
hundreds of millions and a set of Python ints would cost roughly 10x the memory.
"""
import json, os, time

import numpy as np

DST = "data/final_canonical/webqsp/_acquisition/nsm/extracted"
OUT = "data/final_canonical/webqsp/NSM_UNION_RESULT.json"
DS = {"webqsp": f"{DST}/webqsp/webqsp", "CWQ": f"{DST}/CWQ/CWQ"}
SPLITS = ("train", "dev", "test")

TARGET_TRIPLES = 8309195
TARGET_RELATIONS = 7058
BAND = 0.10                       # pre-registered, not adjustable here

EP_BITS, REL_BITS = 23, 14        # 8,388,608 endpoints / 16,384 relations
EP_MASK, REL_MASK = (1 << EP_BITS) - 1, (1 << REL_BITS) - 1
CHUNK_Q = 2000                    # questions per np.unique pass
CONSOLIDATE_EVERY = 12            # chunk-unique arrays to hold before collapsing


def read_vocab(p):
    with open(p, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh]


def main():
    t0 = time.time()
    gep, grel = {}, {}                 # global string -> global id
    maps = {}
    vocab_stats = {}
    for ds, d in DS.items():
        ents, rels = read_vocab(f"{d}/entities.txt"), read_vocab(f"{d}/relations.txt")
        e_map = np.empty(len(ents), dtype=np.int64)
        for i, s in enumerate(ents):
            g = gep.get(s)
            if g is None:
                g = gep[s] = len(gep)
            e_map[i] = g
        r_map = np.empty(len(rels), dtype=np.int64)
        for i, s in enumerate(rels):
            g = grel.get(s)
            if g is None:
                g = grel[s] = len(grel)
            r_map[i] = g
        maps[ds] = (e_map, r_map)
        vocab_stats[ds] = {"entities_txt_lines": len(ents), "relations_txt_lines": len(rels)}
        print(f"[vocab] {ds} ents={len(ents):,} rels={len(rels):,} "
              f"global_ents={len(gep):,} global_rels={len(grel):,}", flush=True)

    if len(gep) > EP_MASK or len(grel) > REL_MASK:
        raise SystemExit(f"bit budget exceeded: global ents={len(gep)} rels={len(grel)}")

    parts, per_ds, raw_total, q_total = [], {}, 0, 0

    def consolidate():
        if len(parts) > 1:
            u = np.unique(np.concatenate(parts))
            parts.clear()
            parts.append(u)

    for ds, d in DS.items():
        e_map, r_map = maps[ds]
        ds_parts, ds_raw, ds_q = [], 0, 0
        for sp in SPLITS:
            p = f"{d}/{sp}_simple.json"
            if not os.path.exists(p):
                print(f"[skip ] {p} absent", flush=True)
                continue
            buf = []
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    tp = (rec.get("subgraph") or {}).get("tuples") or []
                    ds_q += 1
                    q_total += 1
                    if tp:
                        buf.append(np.asarray(tp, dtype=np.int64))
                    if ds_q % CHUNK_Q == 0 and buf:
                        a = np.concatenate(buf); buf.clear()
                        ds_raw += len(a); raw_total += len(a)
                        codes = ((e_map[a[:, 0]] << (REL_BITS + EP_BITS))
                                 | (r_map[a[:, 1]] << EP_BITS) | e_map[a[:, 2]])
                        u = np.unique(codes)
                        parts.append(u); ds_parts.append(u)
                        if len(parts) >= CONSOLIDATE_EVERY:
                            consolidate()
                        print(f"[{ds}/{sp}] q={ds_q:,} raw={ds_raw:,} "
                              f"chunks={len(parts)} t={time.time() - t0:.0f}s", flush=True)
            if buf:
                a = np.concatenate(buf); buf.clear()
                ds_raw += len(a); raw_total += len(a)
                codes = ((e_map[a[:, 0]] << (REL_BITS + EP_BITS))
                         | (r_map[a[:, 1]] << EP_BITS) | e_map[a[:, 2]])
                u = np.unique(codes)
                parts.append(u); ds_parts.append(u)
            print(f"[{ds}/{sp}] DONE q={ds_q:,} raw={ds_raw:,} t={time.time() - t0:.0f}s", flush=True)
        ds_u = np.unique(np.concatenate(ds_parts)) if ds_parts else np.empty(0, dtype=np.int64)
        per_ds[ds] = {**vocab_stats[ds], "questions": ds_q, "raw_tuples": ds_raw,
                      "distinct_triples": int(ds_u.size),
                      "distinct_endpoints": int(np.union1d(ds_u >> (REL_BITS + EP_BITS),
                                                           ds_u & EP_MASK).size),
                      "distinct_relations_used": int(np.unique((ds_u >> EP_BITS) & REL_MASK).size)}
        del ds_parts, ds_u
        consolidate()

    codes = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
    n_tri = int(codes.size)
    n_rel = int(np.unique((codes >> EP_BITS) & REL_MASK).size)
    n_ep = int(np.union1d(codes >> (REL_BITS + EP_BITS), codes & EP_MASK).size)

    d_tri = abs(n_tri - TARGET_TRIPLES) / TARGET_TRIPLES
    d_rel = abs(n_rel - TARGET_RELATIONS) / TARGET_RELATIONS
    if n_tri == TARGET_TRIPLES and n_rel == TARGET_RELATIONS:
        verdict, vid = ("RESOLVED. Artifact is the MID preimage; RoG surfaces become a derived "
                        "display field. The 37-question split delta must still be reported."), "EXACT_PREIMAGE"
    elif d_tri <= BAND and d_rel <= BAND:
        verdict, vid = ("NOT RESOLVED. It is a sibling extraction and must be named WEBQSP_NSM_MID, "
                        "never presented as the RoG graph. Report the delta."), "SIBLING_EXTRACTION"
    else:
        verdict, vid = ("Route A refuted; only route B (Freebase backend re-extraction, = V3) remains."), "REFUTED"

    doc = {
        "schema": "NSM_UNION_RESULT/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preregistration": "data/final_canonical/webqsp/NSM_ACQUISITION_PREREGISTRATION.json",
        "provenance": "data/final_canonical/webqsp/NSM_ACQUISITION_PROVENANCE.json",
        "mid_preservation": "data/final_canonical/webqsp/NSM_MID_PRESERVATION.json",
        "measured": {"triples": n_tri, "relations_used": n_rel, "endpoints": n_ep,
                     "questions": q_total, "raw_tuple_occurrences": raw_total,
                     "global_entity_vocab": len(gep), "global_relation_vocab": len(grel)},
        "targets": {"triples": TARGET_TRIPLES, "relations": TARGET_RELATIONS,
                    "rog_endpoints_reference": 2592894},
        "deltas": {"triples_abs": n_tri - TARGET_TRIPLES, "triples_rel": round(d_tri, 6),
                   "relations_abs": n_rel - TARGET_RELATIONS, "relations_rel": round(d_rel, 6),
                   "endpoints_vs_rog": n_ep - 2592894},
        "band_preregistered": BAND,
        "per_dataset": per_ds,
        "VERDICT_ID": vid,
        "VERDICT": verdict,
        "v1_corpus_changes": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k: doc[k] for k in ("measured", "targets", "deltas", "per_dataset",
                                          "VERDICT_ID", "VERDICT", "elapsed_s")}, indent=1))


if __name__ == "__main__":
    main()
