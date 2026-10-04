"""SECTION E (part 2) -- exact per-source reuse numbers for the deliverable table, plus an
exact bound on the temp-file escaping artifact noticed in part 1.

Part 1 measured the UNION and the webqsp/cwq-only slices. The table also needs RoG-CWQ as a
whole (its entity set overlaps webqsp's), and an exact count of union_entities.txt lines whose
escape does not round-trip (those are the only rows whose measured text could be wrong).
"""
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = "C:/Users/Swastik/Desktop/CRAG"
A = ROOT + "/scratchpad/final_canonical_build/_audit"
LEG_SRC = ROOT + "/data/canonical/webqsp/encodings/_src/docs/shard_*.jsonl"
SPLADE_MAX, DENSE_MAX, BATCH, DIG = 256, 32768, 512, 16
_TK = {}


def _snapshot(repo_dir, must_have):
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    return os.path.dirname(sorted(glob.glob(os.path.join(hub, repo_dir, "snapshots", "*", must_have)))[0])


def _init():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _TK["d"] = AutoTokenizer.from_pretrained(
        _snapshot("models--Alibaba-NLP--gte-Qwen2-1.5B-instruct", "tokenizer.json"), local_files_only=True)
    _TK["s"] = AutoTokenizer.from_pretrained(
        _snapshot("models--naver--splade-cocondenser-ensembledistil", "tokenizer.json"), local_files_only=True)


def _work(args):
    lo, texts = args
    d = np.zeros((len(texts), DIG), np.uint8)
    s = np.zeros((len(texts), DIG), np.uint8)
    for i in range(0, len(texts), BATCH):
        sub = texts[i:i + BATCH]
        di = _TK["d"](sub, add_special_tokens=True, truncation=True, max_length=DENSE_MAX)["input_ids"]
        si = _TK["s"](sub, add_special_tokens=True, truncation=True, max_length=SPLADE_MAX)["input_ids"]
        for k in range(len(sub)):
            d[i + k] = np.frombuffer(hashlib.sha256(",".join(map(str, di[k])).encode()).digest()[:DIG], np.uint8)
            s[i + k] = np.frombuffer(hashlib.sha256(",".join(map(str, si[k])).encode()).digest()[:DIG], np.uint8)
    return lo, d, s


def esc(x):
    return x.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def unesc(x):
    return x.replace("\\r", "\r").replace("\\n", "\n").replace("\\\\", "\\")


def load(p):
    return [unesc(l.rstrip("\n")) for l in open(p, encoding="utf-8")]


def main():
    t0 = time.time()
    # exact escaping round-trip bound
    raw = [l.rstrip("\n") for l in open(A + "/union_entities.txt", encoding="utf-8")]
    bad = sum(1 for r in raw if esc(unesc(r)) != r)
    print(f"[E2] union_entities.txt lines that do NOT round-trip: {bad} of {len(raw):,}", flush=True)

    union = [unesc(r) for r in raw]
    wq = set(load(A + "/webqsp_entities.txt"))
    cq = set(load(A + "/cwq_entities.txt"))
    print(f"[E2] union={len(union):,} webqsp={len(wq):,} cwq={len(cq):,}", flush=True)

    leg = []
    for p in sorted(glob.glob(LEG_SRC)):
        for line in open(p, encoding="utf-8"):
            leg.append(json.loads(line).get("text") or "")
    print(f"[E2] legacy encoder inputs = {len(leg):,}", flush=True)

    distinct = list(dict.fromkeys(union + leg))
    D = np.zeros((len(distinct), DIG), np.uint8)
    S = np.zeros((len(distinct), DIG), np.uint8)
    CH = 20000
    jobs = [(i, distinct[i:i + CH]) for i in range(0, len(distinct), CH)]
    with mp.Pool(5, initializer=_init) as pool:
        for lo, d, s in pool.imap_unordered(_work, jobs, chunksize=1):
            D[lo:lo + len(d)] = d
            S[lo:lo + len(s)] = s
    pos = {t: i for i, t in enumerate(distinct)}
    print(f"[E2] tokenized {len(distinct):,} distinct inputs {time.time()-t0:.0f}s", flush=True)

    li = np.fromiter((pos[t] for t in leg), np.int64, len(leg))
    legD = set(map(bytes, D[li]))
    legS = set(map(bytes, S[li]))

    def reuse(names):
        idx = np.fromiter((pos[t] for t in names), np.int64, len(names))
        return (sum(1 for b in map(bytes, D[idx]) if b in legD),
                sum(1 for b in map(bytes, S[idx]) if b in legS))

    out = {"_escape_roundtrip_failures": bad, "_escape_note":
           "lines whose temp-file escaping is not invertible; the ONLY rows whose measured text "
           "could be wrong. Everything else is exact.", "sources": {}}
    for name, lst in (("ROG_WEBQSP", [e for e in union if e in wq]),
                      ("ROG_CWQ", [e for e in union if e in cq]),
                      ("UNION", union)):
        d, s = reuse(lst)
        out["sources"][name] = {"ENTITY_N": len(lst), "DENSE_REUSE_N": d, "SPLADE_REUSE_N": s,
                                "DENSE_REENCODE_N": len(lst) - d, "SPLADE_REENCODE_N": len(lst) - s}
        print(f"  {name:12s} N={len(lst):>9,} dense_reuse={d:>9,} splade_reuse={s:>9,}", flush=True)

    with open(A + "/E2_persource.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"WROTE {A}/E2_persource.json ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
