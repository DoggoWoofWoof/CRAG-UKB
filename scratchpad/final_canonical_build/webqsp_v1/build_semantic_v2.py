"""CRAG_FREEBASE_SEMANTIC_OVERLAY_V1 -- neighbour-named context, keyed by node_uid.

    PYTHONHASHSEED=0 python .../build_semantic_v2.py --stage A|B|C|D|all

Turns an anonymous mediator into something a retriever can actually embed:

    Marriage - spouse: Barack Obama; spouse: Michelle Obama; from: 1992
    Film performance - actor: Tom Hanks; film: Forrest Gump; character: Forrest Gump

instead of the roles-only form ("Marriage - roles: spouse, spouse, from"), which says what slots
exist but never says who is in them.

SEPARATE ARTIFACT, NOT AN EDIT
  CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 is frozen at manifest hash
  25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865 and is READ here, never
  written. The canonical graph (301,977,131 nodes / 2,062,430,072 edges) is likewise read-only.
  This pass writes only semantic_v2/ and its own scratch.

THE CAP IS DETERMINISTIC, PER THE INSTRUCTION
  Per node: at most 8 outgoing (relation -> neighbour display_name) and at most 4 incoming
  (relation <- neighbour display_name), drawn only from relations that relation_policy calls
  informative. Which 8 is decided by file order, which is fixed, so the artifact is reproducible.
  Identical (relation, name) pairs collapse at render time -- but "spouse: Barack Obama;
  spouse: Michelle Obama" is kept, because a repeated relation with different neighbours is the
  whole content of a mediator.

DIRECTION IS NOT FAKED
  PASS D stores each fact once, and when it flipped an edge it also rewrote the predicate to the
  master property (v3_pass_d.py: `r[fi] = revv[pos[fi]]`). So a stored row always reads correctly
  left to right and `flipped` is provenance about the source, not a rendering instruction.
  Outgoing and incoming are still listed separately, so the text never implies the node holds a
  role it does not.

WHY FOUR STAGES
  The join neighbour_uid -> display_name cannot be done in RAM: 302M names are ~8 GB. It is done
  as a partitioned shuffle instead, and the partition key is free -- the node shards are already
  laid out by `node_uid % 128` (lit_b and obj_b ARE bucket b), so only the 6 subj_ shards need
  repartitioning. Each stage is restartable, so a failure never costs another 2.06B-edge scan.

    A  repartition subj_ overlay names into 128 uid%128 buckets     (~124M rows)
    B  scan 2.06B edges, apply policy + caps, spill by neighbour bucket
    C  per bucket, attach neighbour names, repartition by node range
    D  render one node range at a time, write semantic_v2/
"""
import sys, io, os, json, glob, time, shutil, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relation_policy as POL

V3 = "data/final_canonical/freebase_v3"
OVL = f"{V3}/overlay_v1"
SCR = f"{V3}/_semantic_scratch"
OUT = f"{V3}/semantic_v2"
EDGES = sorted(glob.glob(f"{V3}/canonical/edges/*.parquet"))
CACHE = f"{V3}/_acquisition/_unresolved_population.npz"

NB = 128            # neighbour buckets, == the node-shard layout
NR = 64             # node ranges for the second shuffle
K_OUT, K_IN = 8, 4  # the instructed caps
BATCH = 4_000_000
STRUCT = {"STRUCTURAL_INFERRED", "STRUCTURAL_FALLBACK"}
EXPECT_EDGES = 2_062_430_072
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)


def need_population():
    """The 69,777,967 nodes whose display_name is not a real name -- the ones context helps."""
    if os.path.exists(CACHE):
        return np.load(CACHE)["U"]
    uids = []
    for fp in sorted(glob.glob(f"{OVL}/*.parquet")):
        t = pq.read_table(fp, columns=["node_uid", "display_name_source"])
        s = t["display_name_source"].combine_chunks()
        d = s if isinstance(s.type, pa.DictionaryType) else s.dictionary_encode()
        want = np.array([v in STRUCT for v in d.dictionary.to_pylist()], dtype=bool)
        m = want[d.indices.to_numpy(zero_copy_only=False)]
        if m.any():
            uids.append(t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)[m])
        del t, s, d
    U = np.sort(np.concatenate(uids))
    np.savez(CACHE, U=U, KC=np.zeros(len(U), dtype=np.uint8))
    return U


def relation_codes():
    """rel_uid -> int16 informative code (-1 = DENY), plus the leaf label for each code."""
    t = pq.read_table(f"{V3}/canonical/relations.parquet", columns=["rel_uid", "relation"])
    ru = t["rel_uid"].combine_chunks().to_numpy(zero_copy_only=False)
    rn = t["relation"].to_pylist()
    o = np.argsort(ru, kind="stable")
    ru = ru[o]
    rn = [rn[i] for i in o.tolist()]
    code = np.full(len(ru), -1, dtype=np.int16)
    leaf, full, n = [], [], 0
    for i, nm in enumerate(rn):
        if POL.classify(nm) == "DENY":
            continue
        code[i] = n
        leaf.append(POL.leaf(nm))
        full.append(str(nm))
        n += 1
    assert n < 32767, f"informative relations {n} overflow int16"
    return ru, code, np.array(leaf, dtype=object), full


# ----------------------------------------------------------------- STAGE A: subj name buckets
def stage_a():
    d = f"{SCR}/names_subj"
    os.makedirs(d, exist_ok=True)
    w = {}
    rows = 0
    try:
        for fp in sorted(glob.glob(f"{OVL}/subj_*.parquet")):
            t = pq.read_table(fp, columns=["node_uid", "display_name"])
            u = t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
            b = np.mod(u, NB)
            order = np.argsort(b, kind="stable")
            t = t.take(pa.array(order))
            bs = b[order]
            bounds = np.searchsorted(bs, np.arange(NB + 1))
            for k in range(NB):
                lo, hi = int(bounds[k]), int(bounds[k + 1])
                if lo == hi:
                    continue
                if k not in w:
                    w[k] = pq.ParquetWriter(f"{d}/b{k:03d}.parquet", t.schema,
                                            compression="zstd")
                w[k].write_table(t.slice(lo, hi - lo))
            rows += len(u)
            log(f"A {os.path.basename(fp)} rows={rows:,}")
            del t, u, b, order, bs
    finally:
        for x in w.values():
            x.close()
    log(f"A done: {rows:,} subj names bucketed")
    return rows


# ----------------------------------------------------------------- STAGE B: the 2.06B-edge scan
def stage_b():
    NEED = need_population()
    N = len(NEED)
    log(f"B need={N:,}")
    r_uid, r_code, r_leaf, r_full = relation_codes()
    log(f"B informative relations={len(r_leaf):,} of {len(r_uid):,}")
    with io.open(f"{SCR}/relation_codes.json", "w", encoding="utf-8") as f:
        json.dump({"leaf": r_leaf.tolist(), "full": r_full}, f, ensure_ascii=False)

    cnt = np.zeros((N, 2), dtype=np.uint8)      # [:,0]=outgoing used, [:,1]=incoming used
    d = f"{SCR}/spill"
    os.makedirs(d, exist_ok=True)
    schema = pa.schema([("nidx", pa.int32()), ("slot", pa.uint8()),
                        ("code", pa.int16()), ("nbr", pa.int64())])
    w = {}
    stats = {"seen": 0, "kept": 0, "anon_dropped": 0}

    def take(side, other, rel, incoming):
        """Select up to K roles for need-nodes on this side; spill them by neighbour bucket.

        Edges are partitioned by src, so ALL of a node's outgoing edges arrive in one batch.
        Keeping one row per node per batch would give every mediator exactly one role, so rows are
        ranked WITHIN their node group and a node can fill several slots from a single batch.
        """
        j = np.searchsorted(NEED, side)
        np.clip(j, 0, N - 1, out=j)
        idx = np.flatnonzero(NEED[j] == side)
        if not idx.size:
            return
        # A neighbour that is itself unnamed contributes nothing: an entity linked to its own
        # notable_for mediator would render "notable for: notable_for". NEED is exactly the set of
        # nodes WITHOUT an attested name, so one lookup decides it -- and doing it here rather than
        # at render time stops an anonymous neighbour from consuming one of the node's 8 slots.
        oth = other[idx]
        jo = np.searchsorted(NEED, oth)
        np.clip(jo, 0, N - 1, out=jo)
        anon = NEED[jo] == oth
        stats["anon_dropped"] += int(anon.sum())
        idx = idx[~anon]
        if not idx.size:
            return
        ri = np.searchsorted(r_uid, rel[idx])
        np.clip(ri, 0, len(r_uid) - 1, out=ri)
        ok = (r_uid[ri] == rel[idx]) & (r_code[ri] >= 0)   # unknown rel_uid is never relabelled
        idx, ri = idx[ok], ri[ok]
        if not idx.size:
            return
        c = 1 if incoming else 0
        K = K_IN if incoming else K_OUT
        jj = j[idx]
        free = cnt[jj, c] < K
        idx, ri, jj = idx[free], ri[free], jj[free]
        if not idx.size:
            return
        order = np.argsort(jj, kind="stable")             # stable: file order decides the winners
        idx, ri, jj = idx[order], ri[order], jj[order]
        newg = np.empty(len(jj), dtype=bool)
        newg[0] = True
        newg[1:] = jj[1:] != jj[:-1]
        starts = np.flatnonzero(newg)
        rank = np.arange(len(jj), dtype=np.int64) - np.repeat(
            starts, np.diff(np.append(starts, len(jj))))
        slot = cnt[jj, c].astype(np.int64) + rank
        keep = slot < K
        idx, ri, jj, slot = idx[keep], ri[keep], jj[keep], slot[keep]
        if not idx.size:
            return
        last = np.empty(len(jj), dtype=bool)
        last[-1] = True
        last[:-1] = jj[:-1] != jj[1:]
        cnt[jj[last], c] = (slot[last] + 1).astype(np.uint8)

        nbr = other[idx]
        tab = pa.table({"nidx": pa.array(jj.astype(np.int32)),
                        "slot": pa.array((slot + (128 if incoming else 0)).astype(np.uint8)),
                        "code": pa.array(r_code[ri]),
                        "nbr": pa.array(nbr)}, schema=schema)
        b = np.mod(nbr, NB)
        o2 = np.argsort(b, kind="stable")
        tab = tab.take(pa.array(o2))
        bs = b[o2]
        bounds = np.searchsorted(bs, np.arange(NB + 1))
        for k in range(NB):
            lo, hi = int(bounds[k]), int(bounds[k + 1])
            if lo == hi:
                continue
            if k not in w:
                w[k] = pq.ParquetWriter(f"{d}/b{k:03d}.parquet", schema, compression="zstd")
            w[k].write_table(tab.slice(lo, hi - lo))
        stats["kept"] += len(idx)

    try:
        for fi, fp in enumerate(EDGES):
            for b in pq.ParquetFile(fp).iter_batches(batch_size=BATCH,
                                                     columns=["src", "rel", "dst"]):
                s = b.column("src").to_numpy(zero_copy_only=False)
                r = b.column("rel").to_numpy(zero_copy_only=False)
                dd = b.column("dst").to_numpy(zero_copy_only=False)
                stats["seen"] += len(s)
                take(s, dd, r, False)
                take(dd, s, r, True)
                del s, r, dd, b
            if fi % 10 == 0 or fi == len(EDGES) - 1:
                log(f"B [{fi+1:3d}/{len(EDGES)}] edges={stats['seen']:,} "
                    f"kept={stats['kept']:,} filled={int((cnt.max(axis=1)>0).sum()):,}")
    finally:
        for x in w.values():
            x.close()

    rec = {"edges_scanned": stats["seen"], "edges_expected": EXPECT_EDGES,
           "EDGES_COMPLETE": stats["seen"] == EXPECT_EDGES,
           "need_nodes": int(N), "slots_selected": stats["kept"],
           "candidate_slots_dropped_anonymous_neighbour": stats["anon_dropped"],
           "nodes_with_at_least_one": int((cnt.max(axis=1) > 0).sum()),
           "nodes_with_outgoing": int((cnt[:, 0] > 0).sum()),
           "nodes_with_incoming": int((cnt[:, 1] > 0).sum()),
           "K_OUT": K_OUT, "K_IN": K_IN}
    with io.open(f"{SCR}/stage_b.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    log(f"B done {rec}")


# ----------------------------------------------------------------- STAGE C: attach names
def stage_c():
    NEED = need_population()
    N = len(NEED)
    span = (N + NR - 1) // NR
    d = f"{SCR}/named"
    os.makedirs(d, exist_ok=True)
    schema = pa.schema([("nidx", pa.int32()), ("slot", pa.uint8()),
                        ("code", pa.int16()), ("name", pa.string())])
    w = {}
    tot = miss = 0
    try:
        for k in range(NB):
            sp = f"{SCR}/spill/b{k:03d}.parquet"
            if not os.path.exists(sp):
                continue
            parts = []
            for pref in ("lit", "obj"):
                fp = f"{OVL}/{pref}_{k:03d}.parquet"
                if os.path.exists(fp):
                    parts.append(pq.read_table(fp, columns=["node_uid", "display_name"]))
            fp = f"{SCR}/names_subj/b{k:03d}.parquet"
            if os.path.exists(fp):
                parts.append(pq.read_table(fp, columns=["node_uid", "display_name"]))
            nt = pa.concat_tables(parts)
            nu = nt["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
            o = np.argsort(nu, kind="stable")
            nu = nu[o]
            nn = nt["display_name"].take(pa.array(o)).combine_chunks()
            del nt, parts

            t = pq.read_table(sp)
            nbr = t["nbr"].combine_chunks().to_numpy(zero_copy_only=False)
            j = np.searchsorted(nu, nbr)
            np.clip(j, 0, len(nu) - 1, out=j)
            hit = nu[j] == nbr
            miss += int((~hit).sum())
            # A neighbour uid that is not in its own bucket would mean the shard layout assumption
            # is wrong, so this is reported rather than silently rendered as a blank.
            names = pc.if_else(pa.array(hit), nn.take(pa.array(j)), pa.scalar(None, pa.string()))
            out = pa.table({"nidx": t["nidx"], "slot": t["slot"],
                            "code": t["code"], "name": names}, schema=schema)
            del t, nu, nn, j, hit, nbr

            ni = out["nidx"].combine_chunks().to_numpy(zero_copy_only=False)
            rg = np.minimum(ni // span, NR - 1)
            o2 = np.argsort(rg, kind="stable")
            out = out.take(pa.array(o2))
            rs = rg[o2]
            bounds = np.searchsorted(rs, np.arange(NR + 1))
            for g in range(NR):
                lo, hi = int(bounds[g]), int(bounds[g + 1])
                if lo == hi:
                    continue
                if g not in w:
                    w[g] = pq.ParquetWriter(f"{d}/r{g:03d}.parquet", schema, compression="zstd")
                w[g].write_table(out.slice(lo, hi - lo))
            tot += out.num_rows
            if k % 16 == 0:
                log(f"C bucket {k}/{NB} rows={tot:,} unresolved_neighbours={miss:,}")
            del out, ni, rg, o2, rs
    finally:
        for x in w.values():
            x.close()
    with io.open(f"{SCR}/stage_c.json", "w", encoding="utf-8") as f:
        json.dump({"rows": tot, "unresolved_neighbour_uids": miss, "span": span}, f, indent=1)
    log(f"C done rows={tot:,} unresolved_neighbours={miss:,}")


# ----------------------------------------------------------------- STAGE D: render
def stage_d():
    NEED = need_population()
    N = len(NEED)
    span = (N + NR - 1) // NR
    with io.open(f"{SCR}/relation_codes.json", encoding="utf-8") as f:
        leaf = json.load(f)["leaf"]
    # display_name for the need-nodes themselves: the head of every rendered string.
    log("D loading need display_names")
    head = np.empty(N, dtype=object)
    for fp in sorted(glob.glob(f"{OVL}/*.parquet")):
        t = pq.read_table(fp, columns=["node_uid", "display_name"])
        u = t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
        j = np.searchsorted(NEED, u)
        np.clip(j, 0, N - 1, out=j)
        hit = np.flatnonzero(NEED[j] == u)
        if hit.size:
            nm = t["display_name"].take(pa.array(hit)).to_pylist()
            head[j[hit]] = nm
        del t, u, j
    os.makedirs(OUT, exist_ok=True)
    written = with_ctx = 0
    for g in range(NR):
        fp = f"{SCR}/named/r{g:03d}.parquet"
        lo = g * span
        hi = min(N, lo + span)
        if hi <= lo:
            continue
        outg = [[] for _ in range(hi - lo)]
        ing = [[] for _ in range(hi - lo)]
        if os.path.exists(fp):
            t = pq.read_table(fp).sort_by([("nidx", "ascending"), ("slot", "ascending")])
            ni = t["nidx"].combine_chunks().to_numpy(zero_copy_only=False)
            sl = t["slot"].combine_chunks().to_numpy(zero_copy_only=False)
            cd = t["code"].combine_chunks().to_numpy(zero_copy_only=False)
            nmv = t["name"].to_pylist()
            for p in range(len(ni)):
                nm = nmv[p]
                if not nm:
                    continue
                bucket = ing if sl[p] >= 128 else outg
                bucket[ni[p] - lo].append(f"{leaf[cd[p]]}: {nm}")
            del t, ni, sl, cd, nmv
        txt = [None] * (hi - lo)
        for i in range(hi - lo):
            o, n = outg[i], ing[i]
            if not o and not n:
                continue
            parts = []
            if o:
                parts.append("; ".join(dict.fromkeys(o)))
            if n:
                parts.append("referenced by - " + "; ".join(dict.fromkeys(n)))
            txt[i] = f"{head[lo+i]} - " + "; ".join(parts)
            with_ctx += 1
        pq.write_table(pa.table({"node_uid": pa.array(NEED[lo:hi], pa.int64()),
                                 "semantic_text": pa.array(txt, pa.string())}),
                       f"{OUT}/part_{g:03d}.parquet", compression="zstd")
        written += hi - lo
        del outg, ing, txt
        if g % 8 == 0:
            log(f"D range {g}/{NR} written={written:,} with_context={with_ctx:,}")
    rec = {"schema": "SEMANTIC_OVERLAY_BUILD/v2",
           "name": "CRAG_FREEBASE_SEMANTIC_OVERLAY_V1",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "APPEND_ONLY": ("keyed by node_uid. canonical/ and overlay_v1/ are read, never written: "
                           "301,977,131 nodes / 2,062,430,072 edges unchanged, RESOLUTION_OVERLAY_V1 "
                           "manifest hash 25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb"
                           "27d865 unchanged."),
           "rows": written, "nodes_with_context": with_ctx,
           "nodes_without_context": written - with_ctx,
           "K_OUT": K_OUT, "K_IN": K_IN, "ranges": NR, "buckets": NB,
           "POPULATION": ("the 69,777,967 nodes whose display_name is not a real name "
                          "(STRUCTURAL_INFERRED or STRUCTURAL_FALLBACK). Nodes that already carry "
                          "an attested name are not in this table; consumers fall back to "
                          "display_name, which for them is the better string."),
           "RENDER": ("<display_name> - <rel>: <neighbour>; ... ; referenced by - <rel>: <neighbour>"),
           "NOT_supported": ["that semantic_text is a name",
                             "that the neighbour list is complete -- it is capped at 8 outgoing "
                             "and 4 incoming informative relations per node"]}
    with io.open(f"{V3}/V3_SEMANTIC_OVERLAY_BUILD.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    log(f"D done rows={written:,} with_context={with_ctx:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
    os.makedirs(SCR, exist_ok=True)
    for st in (["A", "B", "C", "D"] if a.stage == "all" else [a.stage.upper()]):
        log(f"=== STAGE {st} ===")
        {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d}[st]()
