"""Phase 3 (Track B) for the two 5-6M-node corpora: exact per-node artifact REUSE accounting, decided by
FROZEN-TOKENIZER TOKEN IDs -- same contract as reuse_map.py, but streaming + parallel so it fits a 16 GB box.

    python scratchpad/final_canonical_build/reuse_map_kb.py [2wiki|hotpotqa ...] [--workers N] [--verify K]

WHY A SECOND MODULE
    reuse_map.py holds three whole hash tables in RAM (canonical + UKB + Phase-C).  At 6e6 nodes that is several
    GB and ~3.5 h of single-threaded tokenization per corpus.  This module computes the SAME quantity with the
    same decision rule, but:
      * tokenizes each DISTINCT encoder input once (reuse_map.py's `memo`, made explicit and put on disk),
      * parallelises the Phase-C side over its already-sharded encoder-input files,
      * keeps the 6e6 token-ID digests in a memory-mapped .npy instead of a dict.

THE DECISION RULE IS UNCHANGED
    A node's dense/SPLADE row is reusable IFF the frozen tokenizer's exact token-ID sequence for its encoder
    input equals that of an already-encoded row.  Raw-text equality is RECORDED (`status`) but is never the
    decision.  Where two encoder inputs are byte-identical their token-ID hashes are computed once and shared --
    that is memoization of a deterministic pure function, not a text-equality shortcut; --verify K re-tokenizes
    K random shared inputs independently and asserts the digests match, and the result is recorded.

ALREADY-ENCODED ROW FAMILIES CHECKED
    2wiki     PHASE_C_UNIVERSE  data/canonical/2wiki_universe/encodings/{dense,splade}/docs  (5,989,847 rows,
                                both manifests complete=true), encoder input = encodings/_src/docs/shard_*.jsonl
              PHASE_C_VIEW398   data/canonical/2wiki/encodings/...                           (398,354 rows)
              UKB               data/ukb_storage/2wiki_clean/gte_qwen/nodes.npy
    hotpotqa  PHASE_C           data/canonical/hotpotqa/encodings/{dense,splade}/docs         (5,233,329 rows)
              UKB               data/ukb_storage/hotpotqa_clean/gte_qwen/nodes.npy

OUTPUT (identical schema to reuse_map.py)
    data/final_canonical/<ds>/reuse_map/shard_NNN.jsonl + reuse_map/index.json
"""
import os, sys, json, hashlib, collections, time, glob, argparse, multiprocessing as mp
import numpy as np

ROOT = "data/final_canonical"
WORKD = f"{ROOT}/_work/reuse_kb"
SHARD = 250_000
GTE = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
SPLADE = "naver/splade-cocondenser-ensembledistil"
SPLADE_MAX = 256
DENSE_MAX = 32768
BATCH = 256

LEGACY = {"2wiki": "2wiki_clean", "hotpotqa": "hotpotqa_clean"}
NER_STATE = {"2wiki": ("PER_DOC_ON_DISK", "data/canonical/2wiki_universe/_ner_work/ (full-universe per-doc entity postings)"),
             "hotpotqa": ("PER_DOC_ON_DISK", "data/canonical/hotpotqa/_ner_work/")}

# per-dataset already-encoded families: (name, encoder-input shard glob, encodings dir, id -> curid parser)
FAMILIES = {
    "2wiki": [
        ("PHASE_C_UNIVERSE", "data/canonical/2wiki_universe/encodings/_src/docs/shard_*.jsonl",
         "data/canonical/2wiki_universe/encodings", lambda s: s.split(":", 1)[1]),
        ("PHASE_C_VIEW398", "data/canonical/2wiki/encodings/_src/docs/shard_*.jsonl",
         "data/canonical/2wiki/encodings", None),          # keyed by title, not curid -> text-hash family only
    ],
    "hotpotqa": [
        ("PHASE_C", "data/canonical/hotpotqa/encodings/_src/docs/shard_*.jsonl",
         "data/canonical/hotpotqa/encodings", lambda s: s.split("_", 1)[1]),
    ],
}
SHARD_SIZE_SRC = 40_000    # _srcmeta.json shard_size for every encodings/_src/docs family


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def iter_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _snapshot(repo_dir, must_have):
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hits = sorted(glob.glob(os.path.join(hub, repo_dir, "snapshots", "*", must_have)))
    if not hits:
        raise FileNotFoundError(f"no cached snapshot of {repo_dir} containing {must_have}")
    return os.path.dirname(hits[0])


# ------------------------------------------------------------------ worker (one process per shard file)
_TK = {}


def _init():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"      # we parallelise at the process level
    from transformers import AutoTokenizer
    _TK["d"] = AutoTokenizer.from_pretrained(_snapshot("models--Alibaba-NLP--gte-Qwen2-1.5B-instruct", "tokenizer.json"),
                                             local_files_only=True)
    _TK["s"] = AutoTokenizer.from_pretrained(_snapshot("models--naver--splade-cocondenser-ensembledistil", "tokenizer.json"),
                                             local_files_only=True)


def _hash_texts(texts):
    """-> (dense_digests uint8[n,32], splade_digests uint8[n,32])"""
    d = np.zeros((len(texts), 32), np.uint8)
    s = np.zeros((len(texts), 32), np.uint8)
    for i in range(0, len(texts), BATCH):
        sub = texts[i:i + BATCH]
        di = _TK["d"](sub, add_special_tokens=True, truncation=True, max_length=DENSE_MAX)["input_ids"]
        si = _TK["s"](sub, add_special_tokens=True, truncation=True, max_length=SPLADE_MAX)["input_ids"]
        for k in range(len(sub)):
            d[i + k] = np.frombuffer(hashlib.sha256(",".join(map(str, di[k])).encode()).digest(), np.uint8)
            s[i + k] = np.frombuffer(hashlib.sha256(",".join(map(str, si[k])).encode()).digest(), np.uint8)
    return d, s


def work_shard(args):
    """Tokenize one encoder-input shard file; write curid/text-sha/dense/splade digests next to it in _work."""
    src, out_npz = args
    if os.path.exists(out_npz):
        return out_npz, int(np.load(out_npz)["n"][0])
    ids, texts = [], []
    for r in iter_jsonl(src):
        ids.append(r["id"]); texts.append(r.get("text") or "")
    d, s = _hash_texts(texts)
    t = np.frombuffer(b"".join(hashlib.sha256(x.encode("utf-8")).digest() for x in texts),
                      np.uint8).reshape(len(texts), 32)
    np.savez(out_npz, ids=np.array(ids, dtype="S32"), tsha=t, dense=d, splade=s, n=np.array([len(texts)]))
    return out_npz, len(texts)


def work_texts(args):
    """Tokenize an explicit list of texts (canonical rows with no byte-identical already-encoded input)."""
    idx, texts, out_npz = args
    if os.path.exists(out_npz):
        return out_npz
    d, s = _hash_texts(texts)
    np.savez(out_npz, idx=np.array(idx, np.int64), dense=d, splade=s)
    return out_npz


def work_textfile(args):
    """Same, but the (row index, text) pairs are read from a spill file so the parent never holds 6e6 strings."""
    src, out_npz = args
    if os.path.exists(out_npz):
        return out_npz
    idx, texts = [], []
    for r in iter_jsonl(src):
        idx.append(r["i"]); texts.append(r["t"])
    d, s = _hash_texts(texts)
    np.savez(out_npz, idx=np.array(idx, np.int64), dense=d, splade=s)
    return out_npz


def spill_texts(nodes_path, want_mask, prefix, nfiles):
    """Stream nodes.jsonl and round-robin the wanted rows into nfiles spill files. Returns the file list.
    Keeps parent RSS flat: no dict of 6e6 texts, which is what an all-rows-miss would otherwise cost."""
    fhs = [open(f"{prefix}_{k}.jsonl", "w", encoding="utf-8", newline="\n") for k in range(nfiles)]
    n = 0
    try:
        with open(nodes_path, encoding="utf-8") as f:
            for j, line in enumerate(f):
                if not want_mask[j]:
                    continue
                fhs[n % nfiles].write(json.dumps({"i": j, "t": json.loads(line).get("text") or ""},
                                                 ensure_ascii=False) + "\n")
                n += 1
    finally:
        for fh in fhs: fh.close()
    return [f"{prefix}_{k}.jsonl" for k in range(nfiles)], n


# ------------------------------------------------------------------ helpers
KEY = np.dtype([("a", ">u8"), ("b", ">u8")])     # 128-bit prefix of a sha256, lexicographically comparable


def key128(dig):
    """uint8[n,32] -> structured 128-bit key array (first 16 bytes), sortable and searchsortable by numpy."""
    return np.ascontiguousarray(dig[:, :16]).view(KEY).reshape(-1)


class Index:
    """Sorted 128-bit token-ID-digest index; find(q) -> row index or -1, fully vectorised."""

    def __init__(self, dig):
        k = key128(dig)
        self.order = np.argsort(k, kind="stable")
        self.sk = k[self.order]

    def find(self, q):
        n = len(self.sk)
        pos = np.searchsorted(self.sk, q, "left")
        posc = np.clip(pos, 0, max(n - 1, 0))
        ok = (pos < n) & (self.sk[posc] == q)
        out = np.full(len(q), -1, np.int64)
        out[ok] = self.order[posc[ok]]
        return out


def as_int_ids(arr, parse):
    """bytes ids -> int64 curids (-1 where not numeric)."""
    o = np.empty(len(arr), np.int64)
    for i, x in enumerate(arr):
        try:
            o[i] = int(parse(x.decode("utf-8") if isinstance(x, bytes) else x))
        except Exception:
            o[i] = -1
    return o


def hexs(dig_row):
    return dig_row.tobytes().hex()


# ------------------------------------------------------------------ main per-dataset run
def run(ds, workers, verify_k):
    t0 = time.time()
    d = f"{ROOT}/{ds}"
    W = f"{WORKD}/{ds}"
    os.makedirs(W, exist_ok=True)
    # PURGE THE CANONICAL-SIDE CACHES (2026-09-05).  work_textfile/work_texts return an existing .npz untouched,
    # and the step-4 cleanup below removed only the .jsonl spill files, so `spill_todo_*.npz` / `verify_*.npz`
    # survived a run.  Those digests are keyed to the CANONICAL text, which changes whenever textualization does
    # (e.g. TEXTUALIZATION_REV 2 rewrote 94 hotpot rows) -- reusing them would silently report the PREVIOUS
    # revision's reuse counts as if they were measured.  The family caches (`<FAM>_*.npz`) are deliberately kept:
    # they are keyed to the already-encoded encoder-input shards, which are immutable by definition.
    for _stale in glob.glob(f"{W}/spill_todo*") + glob.glob(f"{W}/verify_*.npz"):
        os.remove(_stale)
    out = f"{d}/reuse_map"
    os.makedirs(out, exist_ok=True)

    # ---------- 1. tokenize every already-encoded family's encoder inputs (parallel over its shards) ----------
    fam = {}
    pool = mp.Pool(workers, initializer=_init)
    try:
        for name, pattern, encdir, parse in FAMILIES[ds]:
            shards = sorted(glob.glob(pattern))
            if not shards:
                print(f"  [{ds}] family {name}: NO encoder-input shards at {pattern} -- skipped"); continue
            jobs = [(p, f"{W}/{name}_{os.path.basename(p)}.npz") for p in shards]
            done = 0
            outs = []
            for op, n in pool.imap(work_shard, jobs):
                outs.append(op); done += n
                if len(outs) % 25 == 0:
                    print(f"  [{ds}] {name}: {len(outs)}/{len(shards)} shards, {done} rows, {time.time()-t0:.0f}s", flush=True)
            outs.sort()
            # concatenate into memmapable arrays in shard order (== documents.jsonl / encoding row order)
            ids_all, t_all, d_all, s_all = [], [], [], []
            for op in [f"{W}/{name}_{os.path.basename(p)}.npz" for p in shards]:
                z = np.load(op, allow_pickle=True)
                ids_all.append(z["ids"]); t_all.append(z["tsha"]); d_all.append(z["dense"]); s_all.append(z["splade"])
            ids_all = np.concatenate(ids_all); t_all = np.concatenate(t_all)
            d_all = np.concatenate(d_all); s_all = np.concatenate(s_all)
            np.save(f"{W}/{name}_dense.npy", d_all); np.save(f"{W}/{name}_splade.npy", s_all)
            np.save(f"{W}/{name}_tsha.npy", t_all)
            enc = {}
            for m in ("dense", "splade"):
                mp_ = f"{encdir}/{m}/docs/manifest.json"
                enc[m] = json.load(open(mp_)) if os.path.exists(mp_) else None
            fam[name] = {"ids": ids_all, "tsha": t_all, "dense": d_all, "splade": s_all,
                         "n": len(ids_all), "encdir": encdir, "parse": parse,
                         "dense_complete": bool(enc["dense"] and enc["dense"].get("complete")),
                         "splade_complete": bool(enc["splade"] and enc["splade"].get("complete")),
                         "manifests": {m: (None if enc[m] is None else {k: v for k, v in enc[m].items() if k != "shards_present"})
                                       for m in ("dense", "splade")}}
            print(f"  [{ds}] family {name}: {len(ids_all)} encoder inputs hashed "
                  f"(dense_complete={fam[name]['dense_complete']} splade_complete={fam[name]['splade_complete']}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # ---------- 2. canonical side: raw text sha + curid, streaming ----------
        can_t = []; can_cur = []
        for r in iter_jsonl(f"{d}/nodes.jsonl"):
            can_t.append(hashlib.sha256((r.get("text") or "").encode("utf-8")).digest())
            try:
                can_cur.append(int(r["source_id"]))
            except Exception:
                can_cur.append(-1)
        n_can = len(can_t)
        can_t = np.frombuffer(b"".join(can_t), np.uint8).reshape(n_can, 32)
        can_cur = np.array(can_cur, np.int64)
        print(f"  [{ds}] canonical nodes streamed: {n_can} ({time.time()-t0:.0f}s)", flush=True)

        # ---------- 3. map canonical -> identity-matched already-encoded row (family with a curid parser) ----
        prim = next((n for n in fam if fam[n]["parse"] is not None), None)
        can_tok_d = np.zeros((n_can, 32), np.uint8)
        can_tok_s = np.zeros((n_can, 32), np.uint8)
        have = np.zeros(n_can, bool)
        pc_row = np.full(n_can, -1, np.int64)
        if prim is not None:
            F = fam[prim]
            pc_cur = as_int_ids(F["ids"], F["parse"])
            order = np.argsort(pc_cur, kind="stable"); sc = pc_cur[order]
            pos = np.searchsorted(sc, can_cur, "left")
            posc = np.clip(pos, 0, max(len(sc) - 1, 0))
            ok = (pos < len(sc)) & (sc[posc] == can_cur) & (can_cur >= 0)
            pc_row[ok] = order[posc[ok]]
            m = pc_row >= 0
            same = np.zeros(n_can, bool)
            same[m] = (can_t[m] == F["tsha"][pc_row[m]]).all(axis=1)
            can_tok_d[same] = F["dense"][pc_row[same]]
            can_tok_s[same] = F["splade"][pc_row[same]]
            have = same
            del order, sc, pos, posc, ok, m
            print(f"  [{ds}] identity join vs {prim}: {int((pc_row >= 0).sum())} matched curids, "
                  f"{int(same.sum())} byte-identical encoder inputs ({time.time()-t0:.0f}s)", flush=True)
        # ids / text-shas of the already-encoded families are only needed for the identity join above;
        # drop them before the two 6e6-row token-ID indexes are built (each costs ~150 MB extra)
        for _f in fam.values():
            _f["ids"] = None; _f["tsha"] = None

        # ---------- 4. tokenize the canonical rows whose input is NOT byte-identical to a stored one ----------
        n_todo = int((~have).sum())
        print(f"  [{ds}] canonical rows needing their own tokenization: {n_todo}", flush=True)
        if n_todo:
            # spill to disk rather than hold the texts in RAM: n_todo can legitimately be the whole corpus
            # (it is, whenever no identity-matched family exists), and 6e6 abstracts do not fit next to the digests.
            nfiles = max(1, min(4 * workers, (n_todo + 19_999) // 20_000))
            files, nsp = spill_texts(f"{d}/nodes.jsonl", ~have, f"{W}/spill_todo", nfiles)
            assert nsp == n_todo, (nsp, n_todo)
            jobs = [(p, p.replace(".jsonl", ".npz")) for p in files]
            for op in pool.imap(work_textfile, jobs):
                z = np.load(op)
                can_tok_d[z["idx"]] = z["dense"]; can_tok_s[z["idx"]] = z["splade"]
                del z
            for p in files:                     # the .npz too -- it is canonical-side, see the purge in run()
                for q in (p, p.replace(".jsonl", ".npz")):
                    if os.path.exists(q): os.remove(q)
            print(f"  [{ds}] extra tokenization done ({time.time()-t0:.0f}s)", flush=True)

        # ---------- 4b. VERIFY the memoization: re-tokenize K shared inputs independently ----------
        verify = {"k": 0, "mismatches": 0, "note": "re-tokenized independently to prove that byte-identical encoder "
                                                   "inputs yield identical token-ID digests (memoization is sound)"}
        shared = np.nonzero(have)[0]
        if verify_k and len(shared):
            rng = np.random.default_rng(7)
            pick = sorted(int(x) for x in rng.choice(shared, size=min(verify_k, len(shared)), replace=False))
            want = set(pick); vt = {}
            for j, r in enumerate(iter_jsonl(f"{d}/nodes.jsonl")):
                if j in want:
                    vt[j] = r.get("text") or ""
            jobs = []
            step = max(1, (len(pick) + workers - 1) // workers)
            for w in range(0, len(pick), step):
                blk = pick[w:w + step]
                jobs.append((blk, [vt[i] for i in blk], f"{W}/verify_{w}.npz"))
            mism = 0
            for op in pool.imap(work_texts, jobs):
                z = np.load(op)
                mism += int((z["dense"] != can_tok_d[z["idx"]]).any(axis=1).sum())
                mism += int((z["splade"] != can_tok_s[z["idx"]]).any(axis=1).sum())
                del z
                os.remove(op)               # canonical-side, must not outlive the run
            verify = {"k": len(pick), "mismatches": mism, "PASS": mism == 0, "note": verify["note"]}
            print(f"  [{ds}] memoization verify: k={len(pick)} mismatches={mism}", flush=True)
    finally:
        pool.close(); pool.join()

    # ---------- 5. global token-ID indexes over every already-encoded family ----------
    q_d = key128(can_tok_d)
    q_s = key128(can_tok_s)
    hit_d = np.full(n_can, -1, np.int64); src_d = np.zeros(n_can, np.int8)
    hit_s = np.full(n_can, -1, np.int64); src_s = np.zeros(n_can, np.int8)
    famlist = [n for n in fam]
    for fi, name in enumerate(famlist, start=1):
        F = fam[name]
        if F["dense_complete"]:
            r = Index(F["dense"]).find(q_d)
            take = (hit_d < 0) & (r >= 0); hit_d[take] = r[take]; src_d[take] = fi
        if F["splade_complete"]:
            r = Index(F["splade"]).find(q_s)
            take = (hit_s < 0) & (r >= 0); hit_s[take] = r[take]; src_s[take] = fi
        print(f"  [{ds}] index {name}: dense_reusable so far {int((hit_d>=0).sum())}, "
              f"splade_reusable so far {int((hit_s>=0).sum())} ({time.time()-t0:.0f}s)", flush=True)

    # ---------- 6. UKB legacy side (text-sha only where the dump has no content) ----------
    L = LEGACY[ds]
    lp = f"{ROOT}/_work/legacy/{L}/docs.jsonl"
    leg = {}
    leg_has_content = False
    if os.path.exists(lp):
        with open(lp, encoding="utf-8") as f:
            leg_has_content = "content" in json.loads(f.readline())
        for i, r in enumerate(iter_jsonl(lp)):
            leg[r["node_id"]] = (i, r["sha_exact"])
    idmap_p = f"{d}/node_id_map_legacy.json"
    inv = {}
    if os.path.exists(idmap_p):
        for lid, cid in json.load(open(idmap_p, encoding="utf-8"))["legacy_to_canonical"].items():
            if cid is not None:
                inv.setdefault(cid, lid)

    # ---------- 7. emit ----------
    ner_state, ner_where = NER_STATE[ds]
    cnt = collections.Counter(); shards = []
    fh = None; written = 0

    def open_shard(k):
        nonlocal fh
        if fh: fh.close()
        p = f"{out}/shard_{k:03d}.jsonl"
        shards.append({"shard": k, "file": f"reuse_map/shard_{k:03d}.jsonl"})
        fh = open(p, "w", encoding="utf-8", newline="\n")

    open_shard(0)
    used_legacy = set()
    for j, node in enumerate(iter_jsonl(f"{d}/nodes.jsonl")):
        nid = node["node_id"]
        rs = can_t[j].tobytes().hex()
        dh = hexs(can_tok_d[j]); sh_ = hexs(can_tok_s[j])
        lid = inv.get(nid)
        lrow = ltxt = None
        if lid is not None and lid in leg:
            used_legacy.add(lid); lrow, ltxt = leg[lid]
            status = "EXACT" if ltxt == rs else "TEXT_CHANGED"
        elif lid is not None:
            status = "TEXT_CHANGED"
        else:
            status = "NEW"
        d_src = f"{famlist[src_d[j]-1]}:{int(hit_d[j])}" if hit_d[j] >= 0 else None
        s_src = f"{famlist[src_s[j]-1]}:{int(hit_s[j])}" if hit_s[j] >= 0 else None
        rec = {"canonical_node_id": nid, "source_identity": node.get("source_id"), "canonical_text_hash": rs,
               "legacy_node_id": lid, "legacy_row_index": lrow, "legacy_text_hash": ltxt,
               "status": status, "dense_input_hash": dh, "splade_input_hash": sh_,
               "dense_reusable": d_src is not None, "splade_reusable": s_src is not None,
               "ner_reusable": bool(ner_state == "PER_DOC_ON_DISK" and pc_row[j] >= 0),
               "dense_reuse_source": d_src, "splade_reuse_source": s_src,
               "phase_c_row_index": int(pc_row[j]) if pc_row[j] >= 0 else None,
               "legacy_dense_input_hash": None, "legacy_splade_input_hash": None,
               "ukb_row_token_identical": (None if ltxt is None else (ltxt == rs) or None),
               "ner_reuse_state": ner_state}
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        written += 1
        cnt[status] += 1
        cnt["dense_reusable" if d_src else "dense_needs_encode"] += 1
        cnt["splade_reusable" if s_src else "splade_needs_encode"] += 1
        if d_src: cnt["dense_from_" + d_src.split(":")[0]] += 1
        if s_src: cnt["splade_from_" + s_src.split(":")[0]] += 1
        if written % SHARD == 0:
            open_shard(written // SHARD)
    for lid, (lrow, ltxt) in leg.items():
        if lid in used_legacy:
            continue
        cnt["LEGACY_ONLY"] += 1
        fh.write(json.dumps({"canonical_node_id": None, "source_identity": None, "canonical_text_hash": None,
                             "legacy_node_id": lid, "legacy_row_index": lrow, "legacy_text_hash": ltxt,
                             "status": "LEGACY_ONLY", "dense_input_hash": None, "splade_input_hash": None,
                             "dense_reusable": False, "splade_reusable": False, "ner_reusable": False,
                             "dense_reuse_source": None, "splade_reuse_source": None, "phase_c_row_index": None,
                             "legacy_dense_input_hash": None, "legacy_splade_input_hash": None,
                             "ukb_row_token_identical": None, "ner_reuse_state": ner_state},
                            ensure_ascii=False, sort_keys=True) + "\n")
        written += 1
    fh.close()

    idx = {"dataset": ds, "n_rows": written, "shard_size": SHARD, "shards": shards,
           "fields": ["canonical_node_id", "source_identity", "canonical_text_hash", "legacy_node_id",
                      "legacy_row_index", "legacy_text_hash", "status", "dense_input_hash",
                      "splade_input_hash", "dense_reusable", "splade_reusable", "ner_reusable"],
           "extra_fields": ["dense_reuse_source", "splade_reuse_source", "phase_c_row_index",
                            "legacy_dense_input_hash", "legacy_splade_input_hash",
                            "ukb_row_token_identical", "ner_reuse_state"],
           "reuse_rule": "token-ID equality under the FROZEN tokenizers, not raw-text equality",
           "memoization": {"what": "each DISTINCT encoder input is tokenized once; canonical rows whose input is "
                                   "byte-identical to a stored encoder input share its token-ID digest",
                           "why_sound": "the tokenizer is a deterministic pure function of the input string",
                           "independently_verified": verify},
           "frozen_dense": {"model": GTE, "input": "text field verbatim (no prefix/title for docs)",
                            "max_seq_length": DENSE_MAX,
                            "source": "src/experiments/canonical_encode.py:176-189 + sentence_bert_config.json"},
           "frozen_splade": {"model": SPLADE, "truncation": True, "max_length": SPLADE_MAX,
                             "source": "src/experiments/canonical_encode.py:209 + encodings/splade/docs/status/shard_*.json"},
           "already_encoded_families": {n: {"n_rows": fam[n]["n"], "encodings": fam[n]["encdir"],
                                            "dense_complete": fam[n]["dense_complete"],
                                            "splade_complete": fam[n]["splade_complete"],
                                            "manifests": fam[n]["manifests"]} for n in fam},
           "legacy_ukb": {"docs_dump": lp if os.path.exists(lp) else None,
                          "n_legacy_docs": len(leg), "dump_has_content": leg_has_content,
                          "embeddings": f"data/ukb_storage/{L}/gte_qwen/nodes.npy",
                          "limitation": None if leg_has_content else
                          "the legacy dump for this substrate stores only text hashes (legacy_dump.KEEP_CONTENT), so a "
                          "UKB row is credited only on EXACT text equality; UKB is a secondary source here because the "
                          "Phase-C family already covers the corpus at full scale."},
           "ner": {"state": ner_state, "where": ner_where},
           "counts": dict(cnt), "seconds": round(time.time() - t0, 1)}
    json.dump(idx, open(f"{out}/index.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  [{ds}] map rows={written} shards={len(shards)} counts={dict(cnt)} ({time.time()-t0:.0f}s)", flush=True)
    return idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", nargs="*", default=["2wiki", "hotpotqa"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--verify", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(WORKD, exist_ok=True)
    allidx = {}
    for ds in a.ds:
        allidx[ds] = run(ds, a.workers, a.verify)
    p = f"{ROOT}/_work/reuse_map_summary.json"
    prev = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    prev.update(allidx)
    json.dump(prev, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("wrote", p)
