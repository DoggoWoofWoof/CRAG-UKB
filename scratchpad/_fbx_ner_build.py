"""FBX_SCALE Track A -- the Freebase NER 'shares-entity' edge family, built as its OWN family (never merged into STRUCT or KNN), out of core, resumable, deterministic.

The recipe is the frozen one (src/pipeline/ner_edges.build_ner_edges), applied to the node NAMES exactly as it is on the two KB substrates (metaqa / webqsp encode bare names):
  one document per node (the name);  spaCy en_core_web_sm, parser + lemmatizer disabled, batch 256;  entities whose label is in ENT_LABELS and len(e.text) > 2;  key = e.text.lower().strip();
  df(key) = number of DISTINCT documents holding the key;  a key with 2 <= df <= 25 links every pair of its documents with weight 1 / df;  a pair's weight is the sum over its shared keys.
Only the bookkeeping differs, because 302M documents do not fit a Python dict: keys are 128-bit blake2b digests of the UTF-8 key (collision odds ~1e-20 for ~1e8 keys), documents are the
Freebase node POSITIONS, and the pair graph is built by external sorting.  _test() checks the whole chain against build_ner_edges on a synthetic corpus: the same pairs, the same weights.

Stages (every output is write-once; a record is written LAST, so a killed unit is simply redone):
  EXTRACT <first> <last> <procs>   units = the parquet row groups of nodes/shard_*.parquet in (shard, row group) order; each unit -> ent/s<shard>g<rg>.{hi,lo,pos,off}.npy + .json
  GROUP                            per hash bucket b (top HB bits of the key): gather the bucket from every unit, sort, df, keep 2 <= df <= 25, emit pairs by u-range -> pairs/g<b>_r<r>.{key,w}.npy
  AGG                              per u-range r: concatenate the pairs of every bucket, sort by key, sum weights -> ner_raw/r<r>.{key,w}.npy (key = u * N + v, u < v, globally sorted over r)
  FINALIZE                         manifest of the family: counts, df histogram, degree summary, weights, SHA256 + bytes of every file (write-once record FBX_NER_FAMILY__v1.json)
  XCHECK                           entity-key hash of 20,000 names (run on the laptop and on the host: the lines must agree)
  TEST                             the equivalence test above incl. FINALIZE (seconds)

Output root: data/freebase_scale/ner/ (bulk, git-ignored); records: results/FREEBASE_SCALE/ner/.  data/final_canonical is never written.
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TREE = os.path.join(REPO, "data", "final_canonical", "freebase")
ROOT = os.path.join(REPO, "data", "freebase_scale", "ner")
RECS = os.path.join(REPO, "results", "FREEBASE_SCALE", "ner")
HB = 6                       # key buckets = 2**HB
NR = 64                      # u-ranges
MAXDF = 25
BATCH = 256
MAXCHARS = 1000000           # spaCy's own nlp.max_length: a name longer than this is cut to its first MAXCHARS characters (1 of 301,977,131 names is; FBX_NAME_TAIL__v1.json); shorter names are untouched


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wj(path, obj):
    assert not os.path.exists(path), "write-once: %s exists" % path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def save_npy(path, a):
    tmp = path + ".tmp.npy"
    np.save(tmp, a)
    os.replace(tmp, path)


def key128(k):
    d = hashlib.blake2b(k.encode("utf-8", "surrogatepass"), digest_size=16).digest()
    return int.from_bytes(d[:8], "little"), int.from_bytes(d[8:], "little")


def units():
    """[(shard, row_group, first_row_in_shard, n_rows)] over the node shards in order; shard files are 4M rows, row groups 500k."""
    import pyarrow.parquet as pq
    out = []
    names = sorted(f for f in os.listdir(os.path.join(TREE, "nodes")) if f.startswith("shard_") and f.endswith(".parquet"))
    for f in names:
        s = int(f[6:11])
        md = pq.ParquetFile(os.path.join(TREE, "nodes", f)).metadata
        r0 = 0
        for g in range(md.num_row_groups):
            n = md.row_group(g).num_rows
            out.append((s, g, r0, n))
            r0 += n
    return out


# ---------------------------------------------------------------------------------------------------------------- the per-document part (the frozen recipe)
def doc_keys(nlp, names, batch=BATCH):
    """one set of entity keys per document, exactly the recipe's (label whitelist, len > 2, lower().strip())."""
    from src.pipeline.ner_edges import ENT_LABELS
    out = []
    assert nlp.max_length >= MAXCHARS
    for doc in nlp.pipe([str(t)[:MAXCHARS] for t in names], batch_size=batch):
        ks = set()
        for e in doc.ents:
            if e.label_ in ENT_LABELS and len(e.text) > 2:
                ks.add(e.text.lower().strip())
        out.append(ks)
    return out


def mentions(keysets, positions):
    """(hi, lo, pos) sorted by (hi, lo, pos): one row per (key, document); positions are int32/int64 document ids."""
    hi, lo, pos = [], [], []
    for ks, p in zip(keysets, positions):
        for k in ks:
            h, l = key128(k)
            hi.append(h)
            lo.append(l)
            pos.append(int(p))
    hi, lo, pos = np.array(hi, np.uint64), np.array(lo, np.uint64), np.array(pos, np.int64)
    o = np.lexsort((pos, lo, hi))
    return hi[o], lo[o], pos[o]


def bucket_offsets(hi, hb):
    b = np.arange(1 << hb, dtype=np.uint64) << np.uint64(64 - hb)
    return np.concatenate([np.searchsorted(hi, b, side="left"), [len(hi)]]).astype(np.int64)


# ---------------------------------------------------------------------------------------------------------------- stage GROUP / AGG kernels
def group_pairs(hi, lo, pos, N, nr, maxdf=MAXDF):
    """hi/lo/pos sorted by (hi, lo, pos) and holding whole keys only.  Returns (df histogram, {r: (keys int64, w float64)}): one row per document pair of a key with 2 <= df <= maxdf,
    key = u * N + v with u < v, w = 1 / df, routed to the u-range r = u * nr // N."""
    n = len(hi)
    if n == 0:
        return {"df1": 0, "df2_25": 0, "df_gt_25": 0, "keys": 0, "mentions": 0, "pairs": 0}, {}
    new = np.ones(n, bool)
    new[1:] = (hi[1:] != hi[:-1]) | (lo[1:] != lo[:-1])
    st = np.flatnonzero(new)
    df = np.diff(np.concatenate([st, [n]]))
    hist = {"df1": int((df == 1).sum()), "df2_25": int(((df >= 2) & (df <= maxdf)).sum()), "df_gt_25": int((df > maxdf).sum()), "keys": int(len(df)), "mentions": int(n)}
    out = {}
    npairs = 0
    for d in range(2, maxdf + 1):
        g = st[df == d]
        if len(g) == 0:
            continue
        M = pos[g[:, None] + np.arange(d)[None, :]]           # (G, d), ascending within a row (pos is the last sort key)
        iu, ju = np.triu_indices(d, 1)
        u = M[:, iu].ravel()
        v = M[:, ju].ravel()
        assert (u < v).all(), "duplicate document inside a key (mentions must be distinct per document)"
        k = u * np.int64(N) + v
        r = (u * nr // N).astype(np.int64)
        w = np.full(len(k), 1.0 / d)
        npairs += len(k)
        for rr in np.unique(r):
            m = r == rr
            out.setdefault(int(rr), []).append((k[m], w[m]))
    hist["pairs"] = int(npairs)
    res = {rr: (np.concatenate([a for a, _ in lst]), np.concatenate([b for _, b in lst])) for rr, lst in out.items()}
    return hist, res


def aggregate(keys, w):
    """sum the weights of equal keys; deterministic (stable sort, weights added in sorted order in float64); returns (unique sorted keys, float32 weights)."""
    if len(keys) == 0:
        return keys, w.astype(np.float32)
    o = np.lexsort((w, keys))                                    # equal keys: ascending weights -> the float64 sum is a function of the multiset
    k, ww = keys[o], w[o]
    new = np.ones(len(k), bool)
    new[1:] = k[1:] != k[:-1]
    st = np.flatnonzero(new)
    return k[st], np.add.reduceat(ww, st).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------- stage EXTRACT
_NLP = None


def _worker_init():
    global _NLP
    from src.pipeline.ner_edges import _nlp
    _NLP = _nlp()


def extract_unit(unit):
    import pyarrow.parquet as pq
    s, g, r0, n = unit
    base = os.path.join(ROOT, "ent", "s%03dg%d" % (s, g))
    rec = base + ".json"
    if os.path.exists(rec):
        return {"unit": [s, g], "skipped": True}
    t0 = time.time()
    tb = pq.ParquetFile(os.path.join(TREE, "nodes", "shard_%05d.parquet" % s)).read_row_group(g, columns=["position", "name"])
    positions = tb.column("position").to_numpy()
    names = tb.column("name").to_pylist()
    assert len(names) == n and len(positions) == n
    ks = doc_keys(_NLP, ["" if x is None else x for x in names])
    hi, lo, pos = mentions(ks, positions)
    off = bucket_offsets(hi, HB)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    files = {}
    for nm, a in (("hi", hi), ("lo", lo), ("pos", pos.astype(np.int32)), ("off", off)):
        p = "%s.%s.npy" % (base, nm)
        save_npy(p, a)
        files[nm] = {"file": os.path.relpath(p, REPO).replace("\\", "/"), "sha256": sha256(p)}
    import spacy
    from src.pipeline.ner_edges import ENT_LABELS
    wj(rec, {"RECORD": "FBX_NER_UNIT", "shard": s, "row_group": g, "rows": n, "position_first": int(positions[0]), "position_last": int(positions[-1]), "documents_with_a_key": int(sum(1 for k in ks if k)), "max_chars_rule": MAXCHARS, "truncated_documents": int(sum(1 for x in names if x is not None and len(str(x)) > MAXCHARS)),
             "mentions": int(len(hi)), "seconds": round(time.time() - t0, 1), "spacy": spacy.__version__, "model": _NLP.meta.get("name") + " " + _NLP.meta.get("version", "?"), "labels": sorted(ENT_LABELS),
             "hash_buckets": 1 << HB, "files": files, "code_sha256": sha256(os.path.abspath(__file__)), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return {"unit": [s, g], "seconds": round(time.time() - t0, 1), "mentions": int(len(hi))}


def cmd_extract(first, last, procs):
    us = units()
    sel = us[first:last + 1]
    print("EXTRACT units %d..%d of %d (%d rows), %d process(es)" % (first, min(last, len(us) - 1), len(us), sum(u[3] for u in sel), procs), flush=True)
    t0 = time.time()
    if procs <= 1:
        _worker_init()
        for u in sel:
            print(extract_unit(u), "%.0fs" % (time.time() - t0), flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(procs, initializer=_worker_init) as pool:
            for r in pool.imap_unordered(extract_unit, sel):
                print(r, "%.0fs" % (time.time() - t0), flush=True)


# ---------------------------------------------------------------------------------------------------------------- stages GROUP / AGG / FINALIZE
def cmd_group(root=ROOT, N=None, hb=HB, nr=NR, unit_names=None):
    N = N or int(np.load(os.path.join(TREE, "nodes", "kind.npy"), mmap_mode="r").shape[0])
    unit_names = unit_names or sorted(f[:-5] for f in os.listdir(os.path.join(root, "ent")) if f.endswith(".json"))
    assert unit_names, "no extracted unit"
    hists = {}
    for b in range(1 << hb):
        rec = os.path.join(RECS if root == ROOT else root, "group_b%02d.json" % b)
        if os.path.exists(rec):
            continue
        t0 = time.time()
        H_, L_, P_ = [], [], []
        for nm in unit_names:
            base = os.path.join(root, "ent", nm)
            off = np.load(base + ".off.npy")
            a, z = int(off[b]), int(off[b + 1])
            if z > a:
                H_.append(np.load(base + ".hi.npy", mmap_mode="r")[a:z])
                L_.append(np.load(base + ".lo.npy", mmap_mode="r")[a:z])
                P_.append(np.load(base + ".pos.npy", mmap_mode="r")[a:z])
        hi = np.concatenate(H_) if H_ else np.zeros(0, np.uint64)
        lo = np.concatenate(L_) if L_ else np.zeros(0, np.uint64)
        pos = np.concatenate(P_).astype(np.int64) if P_ else np.zeros(0, np.int64)
        o = np.lexsort((pos, lo, hi))
        hist, per_r = group_pairs(hi[o], lo[o], pos[o], N, nr)
        files = {}
        for r, (k, w) in per_r.items():
            for nm, a in (("key", k), ("w", w)):
                p = os.path.join(root, "pairs", "g%02d_r%02d.%s.npy" % (b, r, nm))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                save_npy(p, a)
        hist.update({"bucket": b, "seconds": round(time.time() - t0, 1), "ranges_written": sorted(per_r), "units": len(unit_names)})
        wj(rec, dict({"RECORD": "FBX_NER_GROUP_BUCKET"}, **hist))
        hists[b] = hist
        print("bucket %d: %s" % (b, {k: v for k, v in hist.items() if k != "ranges_written"}), flush=True)
    return hists


def cmd_agg(root=ROOT, hb=HB, nr=NR):
    for r in range(nr):
        rec = os.path.join(RECS if root == ROOT else root, "agg_r%02d.json" % r)
        if os.path.exists(rec):
            continue
        t0 = time.time()
        K_, W_ = [], []
        for b in range(1 << hb):
            pk = os.path.join(root, "pairs", "g%02d_r%02d.key.npy" % (b, r))
            if os.path.exists(pk):
                K_.append(np.load(pk))
                W_.append(np.load(pk.replace(".key.", ".w.")))
        keys = np.concatenate(K_) if K_ else np.zeros(0, np.int64)
        w = np.concatenate(W_) if W_ else np.zeros(0, np.float64)
        n_in = len(keys)
        k, ww = aggregate(keys, w)
        os.makedirs(os.path.join(root, "ner_raw"), exist_ok=True)
        pk, pw = os.path.join(root, "ner_raw", "r%02d.key.npy" % r), os.path.join(root, "ner_raw", "r%02d.w.npy" % r)
        save_npy(pk, k)
        save_npy(pw, ww)
        wj(rec, {"RECORD": "FBX_NER_AGG_RANGE", "range": r, "pairs_in": int(n_in), "distinct_pairs": int(len(k)), "seconds": round(time.time() - t0, 1),
                 "files": {"key": {"sha256": sha256(pk), "bytes": os.path.getsize(pk)}, "w": {"sha256": sha256(pw), "bytes": os.path.getsize(pw)}}})
        print("range %d: %d pair rows -> %d distinct pairs (%.0fs)" % (r, n_in, len(k), time.time() - t0), flush=True)


def cmd_finalize(root=ROOT, hb=HB, nr=NR, N=None, out=None):
    """the family manifest: counts, df histogram, degree summary, weights, SHA256 + bytes of every served file; write-once record."""
    N = N or int(np.load(os.path.join(TREE, "nodes", "kind.npy"), mmap_mode="r").shape[0])
    out = out or os.path.join(RECS, "FBX_NER_FAMILY__v1.json")
    assert not os.path.exists(out), "write-once: %s exists" % out
    recdir = RECS if root == ROOT else root
    unit_recs = sorted(f for f in os.listdir(os.path.join(root, "ent")) if f.endswith(".json"))
    us = units() if root == ROOT else None
    if us is not None:
        assert len(unit_recs) == len(us), "EXTRACT incomplete: %d of %d units" % (len(unit_recs), len(us))
    rows = docs_key = mentions = 0
    specs = set()
    for f in unit_recs:
        r = json.load(open(os.path.join(root, "ent", f)))
        rows += r["rows"]
        docs_key += r["documents_with_a_key"]
        mentions += r["mentions"]
        specs.add((r["spacy"], r["model"], tuple(r["labels"]), r["hash_buckets"], r["code_sha256"]))
    if not unit_recs:                                                                            # the synthetic test root holds no unit records
        specs = {("test", "test", (), 1 << hb, "test")}
    assert len(specs) == 1, "units were extracted with different software: %s" % specs
    if us is not None:
        assert rows == N, (rows, N)
    hist = {"keys": 0, "df1": 0, "df2_25": 0, "df_gt_25": 0, "mentions": 0, "pairs": 0}
    for b in range(1 << hb):
        h = json.load(open(os.path.join(recdir, "group_b%02d.json" % b)))
        for k in hist:
            hist[k] += h[k]
    assert not unit_recs or hist["mentions"] == mentions, (hist["mentions"], mentions)
    deg = np.zeros(N, np.int32)
    n_edges = pairs_in = 0
    wsum = 0.0
    wmin, wmax = np.inf, 0.0
    prev = -1
    files = {}
    for r in range(nr):
        pk, pw = os.path.join(root, "ner_raw", "r%02d.key.npy" % r), os.path.join(root, "ner_raw", "r%02d.w.npy" % r)
        k, w = np.load(pk), np.load(pw)
        ar = json.load(open(os.path.join(recdir, "agg_r%02d.json" % r)))
        assert len(k) == ar["distinct_pairs"] and len(w) == len(k), "range %d differs from its record" % r
        pairs_in += ar["pairs_in"]
        if len(k):
            assert (np.diff(k) > 0).all() and k[0] > prev, "keys must be strictly increasing over the whole family"
            prev = int(k[-1])
            u, v = k // np.int64(N), k % np.int64(N)
            assert (u < v).all() and int(v.max()) < N, "a pair must have u < v < N (no self loop, no duplicate orientation)"
            deg += np.bincount(u, minlength=N).astype(np.int32)
            deg += np.bincount(v, minlength=N).astype(np.int32)
            wsum += float(w.astype(np.float64).sum())
            wmin, wmax = min(wmin, float(w.min())), max(wmax, float(w.max()))
        n_edges += len(k)
        files["r%02d" % r] = {"key": {"sha256": ar["files"]["key"]["sha256"], "bytes": ar["files"]["key"]["bytes"]}, "w": {"sha256": ar["files"]["w"]["sha256"], "bytes": ar["files"]["w"]["bytes"]}}
        assert sha256(pk) == ar["files"]["key"]["sha256"] and sha256(pw) == ar["files"]["w"]["sha256"], "range %d file differs from its record" % r
    assert pairs_in == hist["pairs"], "pair rows in AGG differ from GROUP's count"
    nz = deg[deg > 0]
    qs = [50, 90, 99, 99.9]
    dhist = np.bincount(np.floor(np.log2(nz)).astype(np.int64)) if len(nz) else np.zeros(0, np.int64)
    rec = {"RECORD": "FBX_NER_FAMILY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "family": "NER shares-entity (separate from STRUCT and KNN; same global positions)",
           "recipe": "frozen src/pipeline/ner_edges.build_ner_edges applied to the node names: spaCy en_core_web_sm (parser, lemmatizer disabled), ENT_LABELS whitelist, len > 2, key lower().strip(), 2 <= df <= 25, weight sum 1/df",
           "software": {"spacy": sorted(specs)[0][0], "model": sorted(specs)[0][1], "labels": list(sorted(specs)[0][2]), "hash_buckets": sorted(specs)[0][3], "extract_code_sha256": sorted(specs)[0][4]},
           "N": N, "units": len(unit_recs), "rows": rows, "documents_with_a_key": docs_key, "mentions": mentions, "keys": hist["keys"], "keys_df1": hist["df1"], "keys_df_2_25": hist["df2_25"], "keys_df_gt_25": hist["df_gt_25"],
           "pair_rows_before_aggregation": pairs_in, "edges_distinct_undirected": n_edges, "format": "ner_raw/r<NN>.key.npy int64 = u * N + v (u < v, strictly increasing over r and within r) + r<NN>.w.npy float32; %d u-ranges, range of u = u * %d // N" % (nr, nr),
           "weight": {"sum": wsum, "min": wmin if n_edges else None, "max": wmax if n_edges else None, "mean": wsum / n_edges if n_edges else None},
           "degree": {"nodes_with_an_edge": int(len(nz)), "isolated_nodes": int(N - len(nz)), "max": int(nz.max()) if len(nz) else 0, "mean_over_nodes_with_an_edge": float(nz.mean()) if len(nz) else 0.0,
                      "quantiles_over_nodes_with_an_edge": {str(q): float(np.percentile(nz, q)) for q in qs} if len(nz) else {}, "hist_log2": [int(x) for x in dhist]},
           "self_loops": 0, "duplicate_pairs": 0, "sorted_strictly_increasing": True,
           "disk_bytes": {"ner_raw": int(sum(f["key"]["bytes"] + f["w"]["bytes"] for f in files.values()))}, "files": files, "code_sha256": sha256(os.path.abspath(__file__))}
    wj(out, rec)
    print(json.dumps({k: rec[k] for k in ("N", "mentions", "keys", "edges_distinct_undirected", "degree", "weight", "disk_bytes")}, indent=1))
    return rec


def cmd_xcheck(n=20000):
    """platform cross-check: the entity-key sets of the first n names of shard 10, row group 0 (hash of the canonical serialisation); run on the laptop and on the host, the two lines must agree."""
    import pyarrow.parquet as pq
    import spacy
    from src.pipeline.ner_edges import _nlp
    names = pq.ParquetFile(os.path.join(TREE, "nodes", "shard_00010.parquet")).read_row_group(0, columns=["name"]).column("name").to_pylist()[:n]
    ks = doc_keys(_nlp(), ["" if x is None else x for x in names])
    h = hashlib.sha256()
    for k in ks:
        h.update(("\x1f".join(sorted(k)) + "\x1e").encode("utf-8", "surrogatepass"))
    print(json.dumps({"XCHECK": True, "names": len(names), "documents_with_a_key": sum(1 for k in ks if k), "mentions": sum(len(k) for k in ks), "sha256": h.hexdigest(), "spacy": spacy.__version__}))


# ---------------------------------------------------------------------------------------------------------------- the equivalence test
def _test():
    import tempfile
    import scipy.sparse as sp
    from src.pipeline.ner_edges import build_ner_edges, _nlp
    pool = ["Barack Obama", "Michelle Obama", "Paris", "France", "Google", "Microsoft", "Berlin", "Germany", "Albert Einstein", "The Beatles", "John Lennon", "London", "United Kingdom", "Amazon",
            "Toyota", "Tokyo", "Japan", "Nile", "Egypt", "Mount Everest", "Nepal", "Bill Gates", "Apple", "Harvard University", "Stanford University", "Shakespeare", "Hamlet", "Canada", "Toronto", "Mozart"]
    os.environ["CRAG_FORCE_NER"] = "1"                                                          # the frozen builder refuses to run without a cache unless forced; dataset=None writes none
    rng = np.random.RandomState(5)
    texts = []
    for i in range(600):
        m = rng.randint(0, 4)
        texts.append(" ".join(["%s%s" % (pool[j], rng.choice(["", " visited", " and", " in"])) for j in rng.choice(len(pool), m, replace=False)]).strip() or "misc thing %d" % i)
    N = len(texts)
    A = build_ner_edges(None, texts, N, maxdf=25, weighted=True)
    Au = sp.triu(A, k=1).tocoo()
    ref = {(int(a), int(b)): float(v) for a, b, v in zip(Au.row, Au.col, Au.data)}
    assert len(ref) > 100, "the synthetic corpus must exercise the pair graph"
    nlp = _nlp()
    global MAXCHARS
    long_t = "Barack Obama visited Paris and " * 4 + "Tokyo Japan"
    full = doc_keys(nlp, [long_t])
    short_t = "Barack Obama visited Paris"
    short_a = doc_keys(nlp, [short_t])
    mc, MAXCHARS = MAXCHARS, 60                                                                 # the over-length rule: only text beyond MAXCHARS is cut
    try:
        cut = doc_keys(nlp, [long_t])
        assert len(long_t) > 60 and cut == doc_keys(nlp, [long_t[:60]]), "an over-length name is cut to its first MAXCHARS characters"
        assert cut != full and {"tokyo", "japan"} & full[0] and not ({"tokyo", "japan"} & cut[0]), "the cut drops the entities beyond MAXCHARS"
        assert len(short_t) <= 60 and doc_keys(nlp, [short_t]) == short_a, "a short name is untouched"
    finally:
        MAXCHARS = mc
    tmp = tempfile.mkdtemp(prefix="nertest_")
    hb, nr = 3, 5
    ks = doc_keys(nlp, texts)
    names = []
    per = 150                                                                                   # four "units", like row groups
    for u in range(4):
        sl = slice(u * per, (u + 1) * per)
        hi, lo, pos = mentions(ks[sl], np.arange(N)[sl])
        off = bucket_offsets(hi, hb)
        base = os.path.join(tmp, "ent", "u%d" % u)
        os.makedirs(os.path.dirname(base), exist_ok=True)
        for nm, a in (("hi", hi), ("lo", lo), ("pos", pos.astype(np.int32)), ("off", off)):
            np.save("%s.%s.npy" % (base, nm), a)
        names.append("u%d" % u)
        assert ((hi >> np.uint64(64 - hb)) == np.repeat(np.arange(1 << hb), np.diff(off))).all(), "bucket offsets"
    cmd_group(tmp, N, hb, nr, names)
    cmd_agg(tmp, hb, nr)
    got = {}
    prev = -1
    for r in range(nr):
        k = np.load(os.path.join(tmp, "ner_raw", "r%02d.key.npy" % r))
        w = np.load(os.path.join(tmp, "ner_raw", "r%02d.w.npy" % r))
        assert (np.diff(k) > 0).all() and (len(k) == 0 or k[0] > prev), "keys must be strictly increasing across ranges"
        prev = int(k[-1]) if len(k) else prev
        for kk, ww in zip(k, w):
            got[(int(kk) // N, int(kk) % N)] = float(ww)
    assert set(got) == set(ref), (len(got), len(ref), list(set(got) ^ set(ref))[:5])
    assert all(abs(got[k] - ref[k]) <= 1e-6 * max(1.0, abs(ref[k])) for k in ref), "weights differ from build_ner_edges"
    df_hist = {}
    for b in range(1 << hb):
        h = json.load(open(os.path.join(tmp, "group_b%02d.json" % b)))
        for k, v in h.items():
            if k in ("df1", "df2_25", "df_gt_25", "keys", "mentions", "pairs"):
                df_hist[k] = df_hist.get(k, 0) + v
    assert df_hist["pairs"] >= len(ref) and df_hist["mentions"] == sum(len(x) for x in ks)
    # determinism: a second GROUP+AGG on a copy gives byte-identical files
    import shutil
    tmp2 = tempfile.mkdtemp(prefix="nertest2_")
    shutil.copytree(os.path.join(tmp, "ent"), os.path.join(tmp2, "ent"))
    cmd_group(tmp2, N, hb, nr, names)
    cmd_agg(tmp2, hb, nr)
    for r in range(nr):
        for nm in ("key", "w"):
            assert sha256(os.path.join(tmp, "ner_raw", "r%02d.%s.npy" % (r, nm))) == sha256(os.path.join(tmp2, "ner_raw", "r%02d.%s.npy" % (r, nm))), "not deterministic"
    fam = cmd_finalize(tmp, hb, nr, N, out=os.path.join(tmp, "FAMILY.json"))
    assert fam["edges_distinct_undirected"] == len(ref) and fam["degree"]["nodes_with_an_edge"] + fam["degree"]["isolated_nodes"] == N
    dg = np.zeros(N, np.int64)
    for (a, b) in ref:
        dg[a] += 1
        dg[b] += 1
    assert fam["degree"]["max"] == int(dg.max()) and fam["degree"]["nodes_with_an_edge"] == int((dg > 0).sum()), "FINALIZE degree summary"
    assert abs(fam["weight"]["sum"] - sum(ref.values())) <= 1e-4 * sum(ref.values()), "FINALIZE weight sum"
    print("TEST PASS: %d documents, %d distinct pairs equal build_ner_edges (pair set exact, weights <= 1e-6), keys %s; deterministic" % (N, len(ref), df_hist))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["TEST"]:
        _test()
    elif a[:1] == ["EXTRACT"] and len(a) == 4:
        cmd_extract(int(a[1]), int(a[2]), int(a[3]))
    elif a[:1] == ["GROUP"]:
        cmd_group()
    elif a[:1] == ["AGG"]:
        cmd_agg()
    elif a[:1] == ["FINALIZE"]:
        cmd_finalize()
    elif a[:1] == ["XCHECK"]:
        cmd_xcheck()
    else:
        raise SystemExit(__doc__)
