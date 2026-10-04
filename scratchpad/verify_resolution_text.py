"""
COMPLETE RESOLUTION-TEXT GATE
=============================
The strongest statement that can be made about a pointer index without re-encoding 11.4M
vectors:

    for EVERY canonical row i, of EVERY dataset, of BOTH models, the text that actually
    produced the vector the pointer resolves to is the canonical text of node i
    (exactly, or token-identically under that model's frozen tokenizer)

Earlier gates were weaker in two ways this one closes. The text gate only looked at pointers
whose store was PHASE_C, so it silently skipped every row redirected into the dense-repair
store and every row served by the REV2 patch. And it compared against the row named by
phase_c_row_index, which is the node's OWN row -- not the row the reuse decision was made
against. That is what hid three wrong vectors in 2wiki.

Here every store is followed to the text that fed the encoder:

    PHASE_C        text = data/canonical/<tree>/documents.jsonl[row]
    DENSE_REPAIR   text = data/canonical/<tree>/documents.jsonl[dense_rows[row]]
                          (the repair re-encoded Phase-C rows, so a repaired pointer still
                           means "the vector for Phase-C row R" -- only the bytes moved)
    REV2_PATCH     text = _rev2_encoder_patch/encode_worklist/<ds>.jsonl[node_id]
                          and the patch id at that row must be this node

METHOD
    Texts are compared as 128-bit blake2b digests so 11.4M texts cost 182 MB instead of
    tens of GB. A digest collision could only ever hide a mismatch, never invent one, and at
    128 bits over 1.1e7 items that probability is ~1e-25. Every row the digest pass flags is
    then re-read in full and tokenized, because a raw-text difference is not automatically an
    error: reuse is decided by token-ID equality under each frozen tokenizer, and dense
    (gte-Qwen2 @32768) and SPLADE (BERT @256) truncate at very different lengths, so two
    texts differing only past the cut are legitimately the same encoder input.

PASS means: raw-text mismatches are zero, or every one of them is token-identical.
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = "data/final_canonical"
CANON = "data/canonical"
REPAIR = os.path.join(ROOT, "_dense_repair_patch")
DOC_TREE = {"metaqa": "metaqa", "2wiki": "2wiki_universe", "musique": "musique",
            "hotpotqa": "hotpotqa", "squad": "squad"}
MAXLEN = {"dense": 32768, "splade": 256}
TOK = {"dense": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
       "splade": "naver/splade-cocondenser-ensembledistil"}
GTE_QINSTR = ("Instruct: Given a web search query, retrieve relevant passages that answer "
              "the query\nQuery: ")
SPLITS = ["train", "dev", "validation", "test"]
_TOKCACHE = {}


def digests(strings):
    """(N,2) uint64 array of 128-bit blake2b digests, one per string."""
    buf = bytearray()
    n = 0
    for s in strings:
        buf += hashlib.blake2b((s or "").encode("utf-8"), digest_size=16).digest()
        n += 1
    return np.frombuffer(buf, dtype=np.uint64).reshape(n, 2)


def field_stream(path, field):
    with io.open(path, encoding="utf-8") as f:
        for ln in f:
            yield json.loads(ln).get(field) or ""


def line_texts(path, field, rows):
    """Exact texts at the given line numbers, one streaming pass."""
    want = set(int(r) for r in rows)
    out = {}
    if not want:
        return out
    hi = max(want)
    with io.open(path, encoding="utf-8") as f:
        for k, ln in enumerate(f):
            if k in want:
                out[k] = json.loads(ln).get(field) or ""
            if k >= hi:
                break
    return out


def tokenizer(model):
    if model not in _TOKCACHE:
        from transformers import AutoTokenizer
        _TOKCACHE[model] = AutoTokenizer.from_pretrained(TOK[model], trust_remote_code=True)
    return _TOKCACHE[model]


def resolve_phase_c_rows(spec, src, row, tree_of_store):
    """
    Map every pointer to the Phase-C row whose TEXT fed the encoder, or -1 when the vector
    came from the REV2 patch instead. Also returns, per store index, which tree it belongs to
    so multi-tree query channels resolve against the right corpus.
    """
    out = np.full(src.shape, -1, dtype=np.int64)
    patch_mask = np.zeros(src.shape, dtype=bool)
    trees = {}
    for i, s in enumerate(spec):
        m = src == i
        if not m.any():
            continue
        kind = s["store"]
        if kind == "PHASE_C":
            out[m] = row[m]
            trees[i] = s.get("tree", tree_of_store)
        elif kind == "DENSE_REPAIR":
            d = os.path.dirname(s["path"])
            rr = np.asarray(json.load(io.open(os.path.join(d, "dense_rows.json"),
                                              encoding="utf-8")), dtype=np.int64)
            out[m] = rr[row[m]]
            trees[i] = s.get("repairs_tree", tree_of_store)
        elif kind == "REV2_PATCH":
            patch_mask |= m
        else:
            raise SystemExit("unknown store kind %r" % kind)
    return out, patch_mask, trees


def check_docs(ds, canon_h, node_h, res):
    tree = DOC_TREE[ds]
    man = json.load(io.open("%s/POINTER_INDEX.json" % ROOT, encoding="utf-8"))
    pc_path = "%s/%s/documents.jsonl" % (CANON, tree)
    pc_h = digests(field_stream(pc_path, "text"))

    wl_path = "%s/_rev2_encoder_patch/encode_worklist/%s.jsonl" % (ROOT, ds)
    wl = {}
    if os.path.exists(wl_path):
        with io.open(wl_path, encoding="utf-8") as f:
            for ln in f:
                o = json.loads(ln)
                wl[o["node_id"]] = o.get("text") or ""
    wl_h = {k: digests([v])[0] for k, v in wl.items()}

    for model in ("dense", "splade"):
        t0 = time.time()
        spec = man["datasets"][ds][model]["stores"]
        z = np.load("%s/%s/pointer_index/%s.npz" % (ROOT, ds, model))
        src, row = z["src"], z["row"].astype(np.int64)
        pcr, patch_mask, _ = resolve_phase_c_rows(spec, src, row, tree)

        bad_pc, bad_patch, patch_id_bad = [], [], 0
        m = pcr >= 0
        idx = np.nonzero(m)[0]
        if idx.size:
            got = pc_h[pcr[idx]]
            diff = np.nonzero((got != canon_h[idx]).any(axis=1))[0]
            bad_pc = [(int(idx[j]), int(pcr[idx[j]])) for j in diff]

        # patch rows: the patch id at that row must be this node, and the worklist text for
        # that node must be the canonical text
        pidx = np.nonzero(patch_mask)[0]
        if pidx.size:
            pids = json.load(io.open("%s/_rev2_encoder_patch/%s/%s_ids.json"
                                     % (ROOT, ds, model), encoding="utf-8"))
            pid_h = digests(pids)
            got = pid_h[row[pidx]]
            patch_id_bad = int((got != node_h[pidx]).any(axis=1).sum())
            for j in pidx:
                nid = pids[row[j]]
                h = wl_h.get(nid)
                if h is None or (h != canon_h[j]).any():
                    bad_patch.append(int(j))

        r = {"n": int(src.size), "stores": len(spec),
             "from_patch": int(pidx.size), "from_phase_c_or_repair": int(idx.size),
             "patch_id_mismatch": patch_id_bad,
             "raw_text_mismatch_phase_c": len(bad_pc),
             "raw_text_mismatch_patch": len(bad_patch)}

        # every raw mismatch gets re-read in full and tokenized
        if bad_pc:
            npos = [p for p, _ in bad_pc]
            prows = [q for _, q in bad_pc]
            ctx = line_texts("%s/%s/nodes.jsonl" % (ROOT, ds), "text", npos)
            ptx = line_texts(pc_path, "text", prows)
            tk = tokenizer(model)
            ml = MAXLEN[model]
            same, fail = 0, []
            for p, q in bad_pc:
                a = tk(ctx[p], truncation=True, max_length=ml)["input_ids"]
                b = tk(ptx[q], truncation=True, max_length=ml)["input_ids"]
                if a == b:
                    same += 1
                elif len(fail) < 5:
                    fail.append({"canonical_position": p, "phase_c_row": q,
                                 "canonical_text": ctx[p][:120],
                                 "phase_c_text": ptx[q][:120],
                                 "tok_canonical": len(a), "tok_phase_c": len(b)})
            r["token_identical"] = same
            r["token_failures"] = fail
            r["n_token_failures"] = len(bad_pc) - same
        else:
            r["token_identical"] = 0
            r["n_token_failures"] = 0

        r["PASS"] = bool(patch_id_bad == 0 and not bad_patch and r["n_token_failures"] == 0)
        r["seconds"] = round(time.time() - t0, 1)
        res["%s/docs/%s" % (ds, model)] = r
        print("  %-22s n=%-9d patch=%-6d raw_mismatch=%-6d tok_identical=%-6d "
              "tok_fail=%-4d patch_bad=%-4d %s"
              % ("%s/docs/%s" % (ds, model), r["n"], r["from_patch"],
                 r["raw_text_mismatch_phase_c"], r["token_identical"],
                 r["n_token_failures"], patch_id_bad + len(bad_patch),
                 "PASS" if r["PASS"] else "FAIL"), flush=True)
        if not r["PASS"] and r.get("token_failures"):
            print("     ", json.dumps(r["token_failures"][:2])[:500], flush=True)


def check_queries(ds, res):
    man = json.load(io.open("%s/POINTER_INDEX.json" % ROOT, encoding="utf-8"))
    q = man["queries"][ds]
    trees = q["trees"]
    d = "%s/%s/queries/pointer_index" % (ROOT, ds)
    qids = json.load(io.open(os.path.join(d, "query_ids.json"), encoding="utf-8"))

    # canonical question text, in query_ids.json order
    qtext = {}
    for sp in SPLITS:
        p = "%s/%s/queries/%s.jsonl" % (ROOT, ds, sp)
        if not os.path.exists(p):
            continue
        with io.open(p, encoding="utf-8") as f:
            for ln in f:
                o = json.loads(ln)
                qtext[o["query_id"]] = o.get("question") or ""
    canon = [qtext[i] for i in qids]
    canon_h = digests(canon)

    tree_q = {}
    for t in trees:
        tree_q[t] = digests(field_stream("%s/%s/queries.jsonl" % (CANON, t), "question"))
    tree_txt = {t: "%s/%s/queries.jsonl" % (CANON, t) for t in trees}

    for model in ("dense", "splade"):
        t0 = time.time()
        spec = q[model]["stores"]
        z = np.load(os.path.join(d, "%s.npz" % model))
        src, row = z["src"], z["row"].astype(np.int64)
        pcr, patch_mask, tmap = resolve_phase_c_rows(spec, src, row, trees[0])
        if patch_mask.any():
            raise SystemExit("queries should never resolve to a REV2 patch")

        bad = []
        for i, s in enumerate(spec):
            m = (src == i) & (pcr >= 0)
            if not m.any():
                continue
            t = tmap[i]
            idx = np.nonzero(m)[0]
            got = tree_q[t][pcr[idx]]
            diff = np.nonzero((got != canon_h[idx]).any(axis=1))[0]
            bad += [(int(idx[j]), t, int(pcr[idx[j]])) for j in diff]

        r = {"n": int(src.size), "stores": len(spec), "raw_text_mismatch": len(bad)}
        if bad:
            tk = tokenizer(model)
            ml = MAXLEN[model]
            same, fail = 0, []
            by_tree = {}
            for p, t, rr in bad:
                by_tree.setdefault(t, []).append(rr)
            txt = {t: line_texts(tree_txt[t], "question", rs) for t, rs in by_tree.items()}
            for p, t, rr in bad:
                a = tk(GTE_QINSTR + canon[p] if model == "dense" else canon[p],
                       truncation=True, max_length=ml)["input_ids"]
                b = tk(GTE_QINSTR + txt[t][rr] if model == "dense" else txt[t][rr],
                       truncation=True, max_length=ml)["input_ids"]
                if a == b:
                    same += 1
                elif len(fail) < 5:
                    fail.append({"position": p, "tree": t, "row": rr,
                                 "canonical": canon[p][:120], "phase_c": txt[t][rr][:120]})
            r["token_identical"] = same
            r["n_token_failures"] = len(bad) - same
            r["token_failures"] = fail
        else:
            r["token_identical"] = 0
            r["n_token_failures"] = 0
        r["PASS"] = bool(r["n_token_failures"] == 0)
        r["seconds"] = round(time.time() - t0, 1)
        res["%s/queries/%s" % (ds, model)] = r
        print("  %-22s n=%-9d raw_mismatch=%-6d tok_identical=%-6d tok_fail=%-4d %s"
              % ("%s/queries/%s" % (ds, model), r["n"], r["raw_text_mismatch"],
                 r["token_identical"], r["n_token_failures"],
                 "PASS" if r["PASS"] else "FAIL"), flush=True)


def main():
    t0 = time.time()
    which = sys.argv[1:] or ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]
    res = {}
    for ds in which:
        canon_h = digests(field_stream("%s/%s/nodes.jsonl" % (ROOT, ds), "text"))
        node_h = digests(field_stream("%s/%s/nodes.jsonl" % (ROOT, ds), "node_id"))
        check_docs(ds, canon_h, node_h, res)
        del canon_h, node_h
        check_queries(ds, res)
    res["ALL_PASS"] = bool(all(v["PASS"] for k, v in res.items() if isinstance(v, dict)))
    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, io.open("scratchpad/verify_resolution_text.json", "w", encoding="utf-8"),
              indent=1)
    print("\nALL_PASS =", res["ALL_PASS"], " %.1fs" % res["elapsed_s"])


if __name__ == "__main__":
    main()
