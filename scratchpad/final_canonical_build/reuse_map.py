"""Phase 3 (Track B) -- exact per-node artifact REUSE accounting, decided by FROZEN-TOKENIZER TOKEN IDs.

User rule enforced here: "Do not decide reuse from raw-text equality alone."  A node's dense/SPLADE row is
reusable iff the frozen tokenizer's exact token-ID sequence for its encoder input is identical to that of an
already-encoded row.  Raw-text equality is recorded but is never the decision.

Frozen encoders (src/experiments/canonical_encode.py -- the encoder that produced every existing row):
  dense  = Alibaba-NLP/gte-Qwen2-1.5B-instruct via SentenceTransformer, doc input = the text field VERBATIM
           (no prefix, no title; queries get GTE_QINSTR, docs do not); sentence_bert_config max_seq_length
           = 32768, i.e. no truncation at these corpus lengths.  Legacy UKB agrees: doc_prefix "".
  splade = naver/splade-cocondenser-ensembledistil, tokenized with truncation=True, max_length=256
           (verified in data/canonical/<ds>/encodings/splade/docs/status/shard_*.json: "max_len": 256).

Two independent families of ALREADY-COMPUTED rows are checked:
  PHASE_C  data/canonical/<ds>/encodings/{dense,splade}/docs  -- keyed to data/canonical/<ds>/documents.jsonl
           (complete for all six datasets, incl. webqsp 1,316,466 and hotpotqa 5,233,329)
  UKB      data/ukb_storage/<legacy>/gte_qwen/nodes.npy       -- keyed to master_nodes_<legacy>.json order

Output per dataset (sharded at SHARD rows, never downsampled):
  data/final_canonical/<ds>/reuse_map/shard_NNN.jsonl   + reuse_map/index.json
Row fields (exactly the requested names, plus reuse-source provenance):
  canonical_node_id, source_identity, canonical_text_hash,
  legacy_node_id, legacy_row_index, legacy_text_hash,
  status, dense_input_hash, splade_input_hash,
  dense_reusable, splade_reusable, ner_reusable

    python scratchpad/final_canonical_build/reuse_map.py [ds ...]
"""
import os, sys, json, hashlib, collections, time

ROOT = "data/final_canonical"
SHARD = 250_000
GTE = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
SPLADE = "naver/splade-cocondenser-ensembledistil"
SPLADE_MAX = 256
DENSE_MAX = 32768
LEGACY = {"metaqa": "metaqa", "2wiki": "2wiki_clean", "musique": "musique_clean", "squad": "squad_clean",
          "webqsp": "webqsp", "hotpotqa": "hotpotqa_clean"}
# NER: spaCy en_core_web_sm over documents.jsonl text; node-local per-doc output still on disk?
NER_STATE = {"metaqa": ("NOT_APPLICABLE", "MetaQA docs are KB entity names; ner_manifest.ner_available=false"),
             "webqsp": ("NOT_APPLICABLE", "WebQSP docs are Freebase entity names; ner_manifest.ner_available=false"),
             "2wiki": ("PER_DOC_ON_DISK", "data/canonical/2wiki/_ner_work/ (per-shard per-doc entity postings)"),
             "hotpotqa": ("PER_DOC_ON_DISK", "data/canonical/hotpotqa/_ner_work/"),
             "musique": ("EDGES_ONLY", "only graph_ner.tsv persisted; per-doc entities must be re-run (cheap, spaCy sm)"),
             "squad": ("EDGES_ONLY", "only graph_ner.tsv persisted; per-doc entities must be re-run (cheap, spaCy sm)")}
BATCH = 512


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def iter_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _snapshot(repo_dir, must_have):
    """Absolute path of the cached snapshot of `repo_dir` that contains `must_have`."""
    import glob
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hits = sorted(glob.glob(os.path.join(hub, repo_dir, "snapshots", "*", must_have)))
    if not hits:
        raise FileNotFoundError(f"no cached snapshot of {repo_dir} containing {must_have}")
    return os.path.dirname(hits[0])


class Toks:
    """Frozen tokenizers; hashes the exact token-ID sequence the encoder would see. Memoized on raw text sha."""

    def __init__(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        from transformers import AutoTokenizer
        # load from the EXACT local snapshots of the two frozen models (avoids a hub round-trip that the
        # installed transformers makes even under HF_HUB_OFFLINE); paths are recorded in the index.
        self.dense_path = _snapshot("models--Alibaba-NLP--gte-Qwen2-1.5B-instruct", "tokenizer.json")
        self.splade_path = _snapshot("models--naver--splade-cocondenser-ensembledistil", "tokenizer.json")
        self.d = AutoTokenizer.from_pretrained(self.dense_path, local_files_only=True)
        self.s = AutoTokenizer.from_pretrained(self.splade_path, local_files_only=True)
        self.memo = {}
        self.n_tok = 0

    def batch(self, texts):
        """-> list of (raw_sha, dense_input_hash, splade_input_hash) aligned with texts."""
        shas = [sha(t) for t in texts]
        need = [i for i, h in enumerate(shas) if h not in self.memo]
        if need:
            sub = [texts[i] for i in need]
            di = self.d(sub, add_special_tokens=True, truncation=True, max_length=DENSE_MAX)["input_ids"]
            si = self.s(sub, add_special_tokens=True, truncation=True, max_length=SPLADE_MAX)["input_ids"]
            for k, i in enumerate(need):
                self.memo[shas[i]] = (sha(",".join(map(str, di[k]))), sha(",".join(map(str, si[k]))))
            self.n_tok += len(need)
        return [(h, *self.memo[h]) for h in shas]


def legacy_pairs(L):
    """(legacy_node_id, encoder-input text) in master/UKB row order.

    The Phase-0 dump (legacy_dump.py) deliberately omits `content` for metaqa ("a verbalized triple bag,
    not the doc text") -- but that bag IS what the frozen UKB encoder consumed, so it is exactly what a
    token-level reuse test must hash.  Fall back to streaming the master node table for any dataset whose
    dump has no content field.  Read-only over data/processed.
    """
    p = f"{ROOT}/_work/legacy/{L}/docs.jsonl"
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        first = json.loads(f.readline())
    if "content" in first:
        for r in iter_jsonl(p):
            yield r["node_id"], (r.get("content") or "")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from jsonstream import iter_json_array
    mp = f"data/processed/master_nodes_{L}.json"
    for n in iter_json_array(mp):
        if (n.get("metadata") or {}).get("type") != "question":
            yield n["node_id"], n.get("content") or ""


def stream_pairs(tk, pairs, on_row):
    """Tokenize an (id, text) iterable in batches; call on_row(row_index, id, text, raw_sha, dhash, shash)."""
    ids, txt, idxs = [], [], []
    i = 0
    for nid, t in pairs:
        ids.append(nid); txt.append(t); idxs.append(i); i += 1
        if len(ids) >= BATCH:
            for k, (rs, dh, sh_) in enumerate(tk.batch(txt)):
                on_row(idxs[k], ids[k], txt[k], rs, dh, sh_)
            ids, txt, idxs = [], [], []
    if ids:
        for k, (rs, dh, sh_) in enumerate(tk.batch(txt)):
            on_row(idxs[k], ids[k], txt[k], rs, dh, sh_)
    return i


def stream_hashes(tk, path, id_field, text_field, on_row):
    """Stream a jsonl, tokenize in batches, call on_row(row_index, id, text, raw_sha, dhash, shash)."""
    ids, txt, idxs = [], [], []
    i = 0
    for r in iter_jsonl(path):
        ids.append(r[id_field]); txt.append(r.get(text_field) or ""); idxs.append(i); i += 1
        if len(ids) >= BATCH:
            for k, (rs, dh, sh_) in enumerate(tk.batch(txt)):
                on_row(idxs[k], ids[k], txt[k], rs, dh, sh_)
            ids, txt, idxs = [], [], []
    if ids:
        for k, (rs, dh, sh_) in enumerate(tk.batch(txt)):
            on_row(idxs[k], ids[k], txt[k], rs, dh, sh_)
    return i


def run(ds, tk):
    t0 = time.time()
    d = f"{ROOT}/{ds}"
    L = LEGACY[ds]
    out = f"{d}/reuse_map"
    os.makedirs(out, exist_ok=True)

    # ---------- A. canonical nodes ----------
    can = {}          # node_id -> (raw_sha, dense_h, splade_h)
    def onA(i, nid, text, rs, dh, sh_):
        can[nid] = (rs, dh, sh_)
    n_can = stream_hashes(tk, f"{d}/nodes.jsonl", "node_id", "text", onA)
    print(f"  [{ds}] canonical nodes hashed: {n_can} ({time.time()-t0:.0f}s)")

    # ---------- B. legacy (UKB) docs ----------
    lp = f"{ROOT}/_work/legacy/{L}/docs.jsonl"
    leg = {}          # legacy_node_id -> [row_index, raw_sha, dense_h, splade_h]
    leg_by_dense = {}
    leg_by_splade = {}
    n_leg = 0
    if os.path.exists(lp):
        def onB(i, lid, text, rs, dh, sh_):
            leg[lid] = [i, rs, dh, sh_]
            leg_by_dense.setdefault(dh[:24], lid)
            leg_by_splade.setdefault(sh_[:24], lid)
        n_leg = stream_pairs(tk, legacy_pairs(L), onB)
    print(f"  [{ds}] legacy UKB docs hashed: {n_leg} ({time.time()-t0:.0f}s)")

    # ---------- C. Phase-C canonical docs (the rows that actually exist as embeddings) ----------
    pc_p = f"data/canonical/{ds}/documents.jsonl"
    pc_by_dense, pc_by_splade = {}, {}
    n_pc = 0
    pc_enc = {}
    for model in ("dense", "splade"):
        mp = f"data/canonical/{ds}/encodings/{model}/docs/manifest.json"
        pc_enc[model] = json.load(open(mp)) if os.path.exists(mp) else None
    if os.path.exists(pc_p):
        def onC(i, did, text, rs, dh, sh_):
            pc_by_dense.setdefault(dh[:24], (did, i))
            pc_by_splade.setdefault(sh_[:24], (did, i))
        n_pc = stream_hashes(tk, pc_p, "canonical_doc_id", "text", onC)
    print(f"  [{ds}] phase-C docs hashed: {n_pc} ({time.time()-t0:.0f}s)")

    # ---------- D. emit ----------
    inv = {}
    m = json.load(open(f"{d}/node_id_map_legacy.json", encoding="utf-8"))["legacy_to_canonical"]
    for lid, cid in m.items():
        inv.setdefault(cid, lid)
    dense_ok_pc = pc_enc["dense"] and pc_enc["dense"].get("complete")
    splade_ok_pc = pc_enc["splade"] and pc_enc["splade"].get("complete")
    ner_state, ner_where = NER_STATE[ds]

    cnt = collections.Counter()
    shards = []
    fh = None
    written = 0
    def open_shard(k):
        nonlocal fh
        if fh: fh.close()
        p = f"{out}/shard_{k:03d}.jsonl"
        shards.append({"shard": k, "file": f"reuse_map/shard_{k:03d}.jsonl"})
        fh = open(p, "w", encoding="utf-8", newline="\n")
    open_shard(0)

    used_legacy = set()
    for _node in iter_jsonl(f"{d}/nodes.jsonl"):
        nid = _node["node_id"]
        rs, dh, sh_ = can[nid]
        lid = inv.get(nid)
        lrow = ltxt_hash = ldh = lsh = None
        ukb_tok_same = None
        if lid is not None:
            used_legacy.add(lid)
            if lid in leg:
                lrow, ltxt_hash, ldh, lsh = leg[lid]
                # THE decision the user asked for: identical FROZEN-TOKENIZER input despite different raw text
                ukb_tok_same = (ldh == dh)
            # raw-text equality is RECORDED here (status), but it is NOT the reuse decision below
            status = "EXACT" if (lid in leg and ltxt_hash == rs) else "TEXT_CHANGED"
        else:
            status = "NEW"
        # reuse decision = token-id hash equality against any already-encoded row
        d_pc = pc_by_dense.get(dh[:24]) if dense_ok_pc else None
        s_pc = pc_by_splade.get(sh_[:24]) if splade_ok_pc else None
        d_lg = leg_by_dense.get(dh[:24])
        s_lg = leg_by_splade.get(sh_[:24])
        d_src = "PHASE_C:" + d_pc[0] if d_pc else ("UKB:" + d_lg if d_lg else None)
        s_src = "PHASE_C:" + s_pc[0] if s_pc else ("UKB:" + s_lg if s_lg else None)
        ner = (ner_state == "PER_DOC_ON_DISK") and (d_pc is not None)
        rec = {"canonical_node_id": nid, "source_identity": _node.get("source_id"), "canonical_text_hash": rs,
               "legacy_node_id": lid, "legacy_row_index": lrow, "legacy_text_hash": ltxt_hash,
               "status": status, "dense_input_hash": dh, "splade_input_hash": sh_,
               "dense_reusable": d_src is not None, "splade_reusable": s_src is not None,
               "ner_reusable": bool(ner),
               "dense_reuse_source": d_src, "splade_reuse_source": s_src,
               "phase_c_row_index": d_pc[1] if d_pc else None,
               "legacy_dense_input_hash": ldh, "legacy_splade_input_hash": lsh,
               "ukb_row_token_identical": ukb_tok_same,
               "ner_reuse_state": ner_state}
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        written += 1
        cnt[status] += 1
        cnt["dense_reusable" if d_src else "dense_needs_encode"] += 1
        cnt["splade_reusable" if s_src else "splade_needs_encode"] += 1
        if d_src and d_src.startswith("PHASE_C"): cnt["dense_from_PHASE_C"] += 1
        elif d_src: cnt["dense_from_UKB"] += 1
        if ukb_tok_same is True:
            cnt["ukb_row_token_identical"] += 1
            if status == "TEXT_CHANGED":
                cnt["ukb_row_token_identical_DESPITE_text_change"] += 1
        elif ukb_tok_same is False:
            cnt["ukb_row_token_differs"] += 1
            if status == "EXACT":
                cnt["IMPOSSIBLE_text_equal_but_tokens_differ"] += 1
        if written % SHARD == 0:
            open_shard(written // SHARD)

    # LEGACY_ONLY rows (legacy nodes with no canonical counterpart)
    for lid, (lrow, lrs, ldh, lsh) in leg.items():
        if lid in used_legacy:
            continue
        cnt["LEGACY_ONLY"] += 1
        fh.write(json.dumps({"canonical_node_id": None, "source_identity": None, "canonical_text_hash": None,
                             "legacy_node_id": lid, "legacy_row_index": lrow, "legacy_text_hash": lrs,
                             "status": "LEGACY_ONLY", "dense_input_hash": ldh, "splade_input_hash": lsh,
                             "dense_reusable": False, "splade_reusable": False, "ner_reusable": False,
                             "dense_reuse_source": None, "splade_reuse_source": None,
                             "phase_c_row_index": None, "legacy_dense_input_hash": ldh,
                             "legacy_splade_input_hash": lsh, "ukb_row_token_identical": None,
                             "ner_reuse_state": ner_state},
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
           "frozen_dense": {"model": GTE, "input": "text field verbatim (no prefix/title for docs)",
                            "max_seq_length": DENSE_MAX, "source": "src/experiments/canonical_encode.py:176-189 + "
                                                                   "sentence_bert_config.json"},
           "frozen_splade": {"model": SPLADE, "truncation": True, "max_length": SPLADE_MAX,
                             "source": "src/experiments/canonical_encode.py:209 + encodings/splade/docs/status/shard_*.json"},
           "row_sources": {"canonical_nodes": n_can, "legacy_ukb_docs": n_leg, "phase_c_docs": n_pc,
                           "phase_c_dense_complete": bool(dense_ok_pc), "phase_c_splade_complete": bool(splade_ok_pc),
                           "legacy_ukb_embeddings": f"data/ukb_storage/{L}/gte_qwen/nodes.npy",
                           "phase_c_embeddings": f"data/canonical/{ds}/encodings/"},
           "ner": {"state": ner_state, "where": ner_where},
           "counts": dict(cnt), "seconds": round(time.time() - t0, 1),
           "n_distinct_texts_tokenized": tk.n_tok}
    json.dump(idx, open(f"{out}/index.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  [{ds}] map rows={written} shards={len(shards)} counts={dict(cnt)} ({time.time()-t0:.0f}s)")
    tk.memo.clear()
    return idx


if __name__ == "__main__":
    targets = [a for a in sys.argv[1:] if not a.startswith("--")] or ["squad", "metaqa", "musique", "2wiki"]
    tk = Toks()
    allidx = {}
    for ds in targets:
        allidx[ds] = run(ds, tk)
    p = f"{ROOT}/_work/reuse_map_summary.json"
    prev = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    prev.update(allidx)
    json.dump(prev, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("wrote", p)
