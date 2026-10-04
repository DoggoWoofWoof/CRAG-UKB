"""SECTION E -- legacy Phase-C WebQSP  <->  official RoG union: join + TOKEN-ID reuse.

Legacy side  : data/canonical/webqsp/documents.jsonl            (1,316,466 rows)
               data/canonical/webqsp/encodings/_src/docs/shard_*.jsonl  (the EXACT encoder
               inputs that produced the dense + SPLADE stores, both manifests rows_covered
               == n_items == 1,316,466)
Official side: scratchpad/final_canonical_build/_audit/union_entities.txt (2,592,894 entities)
               canonical text under the frozen decision = the RoG surface string VERBATIM
               (_WEBQSP_ACCEPTANCE_GATE.json -> text_representation_decision.canonical_text)

Reuse is decided by FROZEN-TOKENIZER TOKEN-ID EQUALITY, never raw-text equality:
    dense  Alibaba-NLP/gte-Qwen2-1.5B-instruct   add_special_tokens=True truncation max_length=32768
    splade naver/splade-cocondenser-ensembledistil add_special_tokens=True truncation max_length=256
identical to scratchpad/final_canonical_build/reuse_map_kb.py:38-41,100-112.  Each DISTINCT
encoder input is tokenized once (memoization of a deterministic pure function); --verify K
re-tokenizes K random shared inputs in a fresh process and asserts the digests match.

Read-only.  Writes one JSON under scratchpad/final_canonical_build/_audit/.
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter

import numpy as np

ROOT = "C:/Users/Swastik/Desktop/CRAG"
A = ROOT + "/scratchpad/final_canonical_build/_audit"
LEG_DOCS = ROOT + "/data/canonical/webqsp/documents.jsonl"
LEG_SRC = ROOT + "/data/canonical/webqsp/encodings/_src/docs/shard_*.jsonl"
SPLADE_MAX, DENSE_MAX, BATCH = 256, 32768, 512
DIG = 16   # bytes of the token-ID-sequence digest kept (128-bit)

_TK = {}


def _snapshot(repo_dir, must_have):
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hits = sorted(glob.glob(os.path.join(hub, repo_dir, "snapshots", "*", must_have)))
    if not hits:
        raise FileNotFoundError(f"no cached snapshot of {repo_dir} containing {must_have}")
    return os.path.dirname(hits[0])


def _init():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _TK["d"] = AutoTokenizer.from_pretrained(
        _snapshot("models--Alibaba-NLP--gte-Qwen2-1.5B-instruct", "tokenizer.json"), local_files_only=True)
    _TK["s"] = AutoTokenizer.from_pretrained(
        _snapshot("models--naver--splade-cocondenser-ensembledistil", "tokenizer.json"), local_files_only=True)


def _digest(texts):
    d = np.zeros((len(texts), DIG), np.uint8)
    s = np.zeros((len(texts), DIG), np.uint8)
    for i in range(0, len(texts), BATCH):
        sub = texts[i:i + BATCH]
        di = _TK["d"](sub, add_special_tokens=True, truncation=True, max_length=DENSE_MAX)["input_ids"]
        si = _TK["s"](sub, add_special_tokens=True, truncation=True, max_length=SPLADE_MAX)["input_ids"]
        for k in range(len(sub)):
            d[i + k] = np.frombuffer(hashlib.sha256(",".join(map(str, di[k])).encode()).digest()[:DIG], np.uint8)
            s[i + k] = np.frombuffer(hashlib.sha256(",".join(map(str, si[k])).encode()).digest()[:DIG], np.uint8)
    return d, s


def _work(args):
    lo, texts = args
    d, s = _digest(texts)
    return lo, d, s


def unescape(x):
    return x.replace("\\r", "\r").replace("\\n", "\n").replace("\\\\", "\\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 1)))
    ap.add_argument("--verify", type=int, default=500)
    args = ap.parse_args()

    t0 = time.time()
    # ---------------- official side ----------------
    official = [unescape(l.rstrip("\n")) for l in open(A + "/union_entities.txt", encoding="utf-8")]
    ocls = [l.strip() for l in open(A + "/union_entity_class.txt", encoding="utf-8")]
    wq_only = {unescape(l.rstrip("\n")) for l in open(A + "/webqsp_entities.txt", encoding="utf-8")}
    print(f"[E] official union entities = {len(official):,} (webqsp-only slice {len(wq_only):,})", flush=True)

    # ---------------- legacy side ----------------
    leg_text, leg_id, leg_src = {}, {}, {}
    n_leg = 0
    for line in open(LEG_DOCS, encoding="utf-8"):
        r = json.loads(line)
        n_leg += 1
        leg_text[r["original_source_id"]] = r["text"]
        leg_id[r["original_source_id"]] = r["canonical_doc_id"]
        leg_src[r["canonical_doc_id"]] = r["original_source_id"]
    print(f"[E] legacy Phase-C rows = {n_leg:,} distinct endpoints = {len(leg_text):,}", flush=True)

    enc_in, n_enc, enc_mismatch = {}, 0, 0
    for p in sorted(glob.glob(LEG_SRC)):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            n_enc += 1
            ep = leg_src.get(r["id"])
            if ep is None:
                continue
            enc_in[ep] = r.get("text") or ""
            if enc_in[ep] != leg_text[ep]:
                enc_mismatch += 1
    print(f"[E] legacy encoder-input rows = {n_enc:,}  (encoder_input != documents.text in "
          f"{enc_mismatch} rows)", flush=True)

    # ---------------- join + classification ----------------
    off_set = set(official)
    leg_set = set(leg_text)
    inter = off_set & leg_set
    rog_only = len(off_set - leg_set)
    legacy_only = sorted(leg_set - off_set)
    exact_id_text = same_entity_text_changed = same_text_id_changed = 0
    NEWRULE = "webqsp:<sha256(surface)[:24]>"   # build_kb.py:545 candidate canonical id rule
    for e in inter:
        text_same = leg_text[e] == e and enc_in.get(e, leg_text[e]) == e
        id_same = leg_id[e] == "webqsp_ent_" + hashlib.sha256(e.encode("utf-8")).hexdigest()[:16]
        if text_same and id_same:
            exact_id_text += 1
        elif text_same:
            same_text_id_changed += 1
        else:
            same_entity_text_changed += 1
    cls_counts = {
        "EXACT_ID_AND_TEXT": exact_id_text,
        "SAME_ENTITY_TEXT_CHANGED": same_entity_text_changed,
        "SAME_TEXT_ID_CHANGED": same_text_id_changed,
        "ROG_ONLY": rog_only,
        "LEGACY_ONLY": len(legacy_only),
        "AMBIGUOUS_MAPPING": n_leg - len(leg_text),   # >1 legacy row for one endpoint
    }
    print(f"[E] classification {cls_counts}", flush=True)

    # ---------------- token-ID digests over every DISTINCT string ----------------
    distinct = list(dict.fromkeys(official + [enc_in.get(e, leg_text[e]) for e in leg_text]))
    print(f"[E] distinct encoder inputs to tokenize = {len(distinct):,} "
          f"with {args.workers} workers ...", flush=True)
    D = np.zeros((len(distinct), DIG), np.uint8)
    S = np.zeros((len(distinct), DIG), np.uint8)
    CH = 20000
    jobs = [(i, distinct[i:i + CH]) for i in range(0, len(distinct), CH)]
    done = 0
    with mp.Pool(args.workers, initializer=_init) as pool:
        for lo, d, s in pool.imap_unordered(_work, jobs, chunksize=1):
            D[lo:lo + len(d)] = d
            S[lo:lo + len(s)] = s
            done += len(d)
            if done % 200000 < CH:
                print(f"    tokenized {done:,}/{len(distinct):,} {time.time()-t0:.0f}s", flush=True)
    pos = {s: i for i, s in enumerate(distinct)}

    # verification that memoization == independent re-tokenization
    _init()
    rnd = random.Random(0)
    samp = [distinct[rnd.randrange(len(distinct))] for _ in range(min(args.verify, len(distinct)))]
    vd, vs = _digest(samp)
    ok = all(bytes(vd[i]) == bytes(D[pos[samp[i]]]) and bytes(vs[i]) == bytes(S[pos[samp[i]]])
             for i in range(len(samp)))
    print(f"[E] memoization verify on {len(samp)} random inputs: {'PASS' if ok else 'FAIL'}", flush=True)

    # ---------------- reuse decision ----------------
    leg_rows = sorted(leg_text)                       # one already-encoded row per distinct endpoint
    leg_d = {bytes(D[pos[enc_in.get(e, leg_text[e])]]) for e in leg_rows}
    leg_s = {bytes(S[pos[enc_in.get(e, leg_text[e])]]) for e in leg_rows}
    dense_reusable = sum(1 for e in official if bytes(D[pos[e]]) in leg_d)
    splade_reusable = sum(1 for e in official if bytes(S[pos[e]]) in leg_s)
    # webqsp-only slice, for the deliverable table row
    wq_list = [e for e in official if e in wq_only]
    cq_list = [e for e in official if e not in wq_only]
    dense_wq = sum(1 for e in wq_list if bytes(D[pos[e]]) in leg_d)
    splade_wq = sum(1 for e in wq_list if bytes(S[pos[e]]) in leg_s)
    dense_cq = sum(1 for e in cq_list if bytes(D[pos[e]]) in leg_d)
    splade_cq = sum(1 for e in cq_list if bytes(S[pos[e]]) in leg_s)
    # how many stored rows are dead (their token sequence matches no official entity)
    off_d = {bytes(D[pos[e]]) for e in official}
    off_s = {bytes(S[pos[e]]) for e in official}
    dead_d = sum(1 for e in leg_rows if bytes(D[pos[enc_in.get(e, leg_text[e])]]) not in off_d)
    dead_s = sum(1 for e in leg_rows if bytes(S[pos[enc_in.get(e, leg_text[e])]]) not in off_s)
    # raw-text equality, RECORDED but never the decision
    text_eq = len(inter)

    out = {
        "_decision_rule": "token-ID equality under the frozen tokenizers; raw-text equality is recorded "
                          "but is never the decision",
        "_tokenizers": {"dense": "Alibaba-NLP/gte-Qwen2-1.5B-instruct (verbatim text, no prefix, "
                                 f"add_special_tokens=True, truncation, max_length={DENSE_MAX})",
                        "splade": "naver/splade-cocondenser-ensembledistil "
                                  f"(add_special_tokens=True, truncation, max_length={SPLADE_MAX})"},
        "_memoization_verified": ok, "_verify_n": len(samp),
        "legacy": {"rows": n_leg, "distinct_endpoints": len(leg_text),
                   "encoder_input_rows": n_enc,
                   "encoder_input_differs_from_documents_text": enc_mismatch,
                   "dense_store": "data/canonical/webqsp/encodings/dense/docs (rows_covered 1,316,466)",
                   "splade_store": "data/canonical/webqsp/encodings/splade/docs (rows_covered 1,316,466)"},
        "official": {"union_entities": len(official),
                     "webqsp_slice": len(wq_list), "cwq_only_slice": len(cq_list)},
        "classification": cls_counts,
        "classification_notes": {
            "EXACT_ID_AND_TEXT": "endpoint present in both; legacy text AND legacy encoder input are the "
                                 "RoG surface verbatim; legacy id == webqsp_ent_<sha256(surface)[:16]>, "
                                 "i.e. the Phase-C id rule reproduces exactly",
            "SAME_TEXT_ID_CHANGED": f"same text, different id -- this is what happens if the canonical "
                                    f"build adopts {NEWRULE} (build_kb.py:545) instead of the Phase-C rule",
            "LEGACY_ONLY": "legacy rows whose endpoint is NOT a graph endpoint of the official union: "
                           "the gold/topic-injected rows from c1c2_webqsp.py:32-33",
            "AMBIGUOUS_MAPPING": "legacy rows sharing one endpoint (documents.jsonl is 1:1, so 0)"},
        "legacy_only_examples": legacy_only[:10],
        "RAW_TEXT_EQUAL_N": text_eq,
        "DENSE_ROWS_REUSABLE": dense_reusable,
        "DENSE_ROWS_REENCODE": len(official) - dense_reusable,
        "SPLADE_ROWS_REUSABLE": splade_reusable,
        "SPLADE_ROWS_REENCODE": len(official) - splade_reusable,
        "per_slice": {"webqsp_slice": {"N": len(wq_list), "DENSE_REUSABLE": dense_wq,
                                       "SPLADE_REUSABLE": splade_wq},
                      "cwq_only_slice": {"N": len(cq_list), "DENSE_REUSABLE": dense_cq,
                                         "SPLADE_REUSABLE": splade_cq}},
        "stored_rows_not_matching_any_official_entity": {"dense": dead_d, "splade": dead_s},
        "class_mix_of_official_union": dict(Counter(ocls)),
    }
    with open(A + "/E_reuse.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("_")
                      and k not in ("legacy_only_examples", "classification_notes")}, indent=1)[:1600],
          flush=True)
    print(f"WROTE {A}/E_reuse.json  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
