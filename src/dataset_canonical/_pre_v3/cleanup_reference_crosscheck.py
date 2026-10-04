# -*- coding: utf-8 -*-
"""Cross-check every cleanup candidate against everything that could reference it. READ-ONLY.

WHY A SEPARATE PASS
  cleanup_plan.py classifies the two canonical data trees by the package's own dependency
  records, and it is now at 0 reclaimable bytes: everything it can prove dead is gone. What is
  left on the disk (373 GB, 32 GB free) is material the package does not DECLARE either way --
  staging copies, build intermediates, the legacy L1/L2/L3 substrates, Freebase V3 campaign
  caches, git bloat -- and the user's condition for deleting any of it is that nothing
  references it. That condition is checked here, per path, against four independent sources,
  and the verdict is written down BEFORE anything is removed. cleanup_pass_3.py refuses any
  path this record does not mark DELETE.

WHAT COUNTS AS A REFERENCE
  records     every *.json / *.md under data/final_canonical and data/_family_v1 (< 50 MB):
              a path string that starts with the candidate is a citation.
  code        every *.py / *.yaml / *.toml / *.json / *.cfg under src/, configs/, baselines/,
              experiments.py, scratchpad/: a path string that starts with the candidate.
              src/ and configs/ references are LIVE code; scratchpad/ references are the
              historical builders and are reported, not treated as live use.
  results     every *.json / *.md / *.csv under results/ (< 20 MB): a quoted number's
              provenance pointing into a candidate would make it audit material.
  pointers    POINTER_INDEX store paths and the retrieval-cache paths: anything the package
              resolves into.
  by name     records inside data/final_canonical/freebase_v3 cite their inputs RELATIVE to
              that directory ('_acquisition/cascade_names.parquet'), and some records cite a
              file by bare name only.  So every candidate is also searched by its
              freebase_v3-relative path and, when the basename is distinctive (>= 10 chars,
              contains '_' or '.'), by basename, with identifier boundaries on both sides.
              A hit counts exactly like a path citation (records, live code, results) and is
              listed under cited_by_name so the reason is inspectable.

  Candidates that live inside data/ukb_storage are ALSO checked by SUBSTRATE NAME (e.g.
  'hotpotqa_clean'), because the legacy loaders build paths from a storage root plus a
  substrate token rather than spelling the path out.

VERDICT RULE
  DELETE  no record citation, no live-code reference, no results reference, nothing resolves
          into it. Scratchpad-only references are allowed and listed (a builder names its own
          output).
  KEEP    anything else, with the referencing files listed so the reason is inspectable.
  The rule is applied per candidate path; a directory candidate is one path.

Run:
  python src/dataset_canonical/cleanup_reference_crosscheck.py [--out=PATH]
"""
import glob
import io
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = "data/final_canonical"
OUT = os.path.join(ROOT, FC, "CLEANUP_REFERENCE_CROSSCHECK.json")

PROTECTED = [
    "data/final_canonical/freebase_v3/_acquisition/raw",
    "data/final_canonical/freebase_v3/_acquisition/idir",
    "data/final_canonical/freebase_v3/_acquisition/facc1",
    "data/final_canonical/freebase_v3/overlay_v1",
    "data/final_canonical/freebase_v3/canonical",
    "data/final_canonical/freebase_v3/inference_overlay_v1",
    "data/final_canonical/freebase_v3/inferred_name_v1",
    "data/final_canonical/freebase_v3/semantic_v1",
]

# bucket -> list of candidate paths (repo-relative, forward slashes); globs allowed
BUCKETS = {
    "B_STAGING_AND_DUPLICATES": [
        "scratchpad/_cache_bundle",
        "scratchpad/_splade_parts/*/*.npz",
        "data/canonical/2wiki_universe/_ner_work",
        "data/canonical/hotpotqa/_ner_work",
        "data/final_canonical/_dense_repair_patch/*__p*",
        "data/final_canonical/_dense_repair_patch/webqsp__docs",
        "metaqa_temp",
        "__pycache__",
        ".pytest_cache",
    ],
    "D_FREEBASE_V3_INTERMEDIATES": [
        "data/final_canonical/freebase_v3/_acquisition/*",
        "data/final_canonical/freebase_v3/idir_compare",
        "data/final_canonical/freebase_v3/pass_a",
        "data/final_canonical/freebase_v3/pass_a_probe",
        "data/final_canonical/freebase_v3/pass_b",
        "data/final_canonical/freebase_v3/pass_c",
        "data/final_canonical/freebase_v3/probe",
        "data/final_canonical/freebase_v3/_cvt_uids_probe.npy",
    ],
    "F_LEGACY_STACK": [
        "data/ukb_storage/*",
        "data/l2_corpus/*",
        "data/processed",
        "data/raw/*",
        "crag_data_backup",
        "checkpoints",
        "tmp",
        "output",
        "transfer",
        "scratchpad/_l1ov", "scratchpad/ablation_qwen", "scratchpad/_l1hu", "scratchpad/_ctrl_cache",
        "scratchpad/_l1ep", "scratchpad/_ta", "scratchpad/_l1kn", "scratchpad/ablation",
        "scratchpad/_b12", "scratchpad/_parity_tmp", "scratchpad/repair_worklist",
        "scratchpad/ner_validate", "scratchpad/_b11", "scratchpad/full_partition",
    ],
    "E_PROVENANCE_KEEP_BY_RULE": [
        "data/canonical/2wiki_universe/documents.jsonl",
        "data/canonical/hotpotqa/documents.jsonl",
        "data/canonical/2wiki_universe/encodings/_src",
        "data/canonical/hotpotqa/encodings/_src",
        "data/_family_v1/webqsp/encodings/_src",
        "data/final_canonical/webqsp/_acquisition",
        "data/_family_v1/webqsp/graph_ner.tsv",
        # per-doc NER postings for webqsp: the same class of artifact that
        # DOWNSTREAM_REBUILD_POLICY.md marks REUSE for 2wiki and hotpotqa (_ner_work/), so it is
        # kept by the same rule even though no record names the webqsp copy
        "data/_family_v1/webqsp/_ner_work",
        "scratchpad/final_canonical_build",
        "data/original",
    ],
}

# Records whose job is to label, plan or enumerate cleanup material. A citation from one of
# these is the labelling itself, not a use -- the same rule CANONICAL_DEDUP applies to itself
# under citations_not_treated_as_use. They are reported separately as labelled_by.
LABELLING_RECORDS = {
    "CANONICAL_DEDUP.json", "CANONICAL_CLEANUP_PLAN.json", "STALE_HASH_SWEEP.json",
    "HANDOFF_READINESS.json", "PACKAGE_BYTES_LEDGER.json",
    "PACKAGE_BYTES_LEDGER_SCOPE_ADDENDUM.json", "CACHE_AND_REPAIR_CHECK.json",
    "CLEANUP_REFERENCE_CROSSCHECK.json", "CLEANUP_PASS_3.json",
}

PATH_RE = re.compile(
    r"(?:data|scratchpad|results|checkpoints|crag_data_backup|metaqa_temp|tmp|output|transfer|"
    r"archive|baselines|__pycache__|\.pytest_cache)(?:/[A-Za-z0-9_.\-*]+)*")


def norm(p):
    return p.replace("\\", "/")


def tree_size(p):
    ap = os.path.join(ROOT, p)
    if os.path.isfile(ap):
        return os.path.getsize(ap), 1
    tot = n = 0
    for dp, _dn, fn in os.walk(ap):
        for f in fn:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
                n += 1
            except OSError:
                pass
    return tot, n


def expand():
    cands = []
    for bucket, pats in BUCKETS.items():
        for pat in pats:
            hits = sorted(norm(os.path.relpath(h, ROOT))
                          for h in glob.glob(os.path.join(ROOT, pat)))
            if not hits and not any(c in pat for c in "*?"):
                hits = [pat]  # keep the candidate so its absence is recorded
            for h in hits:
                if any(h == pr or h.startswith(pr + "/") for pr in PROTECTED):
                    continue
                if h.endswith(".json") and h.startswith("data/final_canonical/freebase_v3/"):
                    continue  # V3 records are never candidates
                cands.append((bucket, h))
    return cands


def iter_files(base, exts, max_bytes):
    for dp, dn, fn in os.walk(os.path.join(ROOT, base)):
        dn[:] = [d for d in dn if d not in (".git", "__pycache__", "recovered_chats")]
        for f in fn:
            if f.lower().endswith(exts):
                p = os.path.join(dp, f)
                try:
                    if os.path.getsize(p) <= max_bytes:
                        yield p
                except OSError:
                    pass


V3 = "data/final_canonical/freebase_v3/"
BACKSLASHES = re.compile(r"\\+")
GENERIC_NAMES = {"probe", "pass_a", "pass_b", "pass_c", "tmp", "output", "transfer", "checkpoints",
                 "buckets", "postings", "raw", "full", "canonical"}


def alt_tokens(cand):
    """strings other than the full repo path under which a record may cite this candidate.
    A generic freebase_v3-relative name (pass_a, probe) is only searched with a trailing '/',
    so prose never matches but 'pass_a/shards/m*_openers.tsv.gz' does."""
    out = set()
    if cand.startswith(V3):
        rel = cand[len(V3):]
        out.add(rel + "/" if rel in GENERIC_NAMES else rel)
    bn = os.path.basename(cand.rstrip("/"))
    if len(bn) >= 10 and ("_" in bn or "." in bn) and bn not in GENERIC_NAMES:
        out.add(bn)
    if len(bn) >= 6 and bn not in GENERIC_NAMES and os.path.isdir(os.path.join(ROOT, cand)):
        out.add(bn + "/")  # a directory cited through one of its files ('mb_hits/part_00000.parquet')
    out |= FILE_TOKENS.get(cand, set())
    return out


NUMBERED_PART = re.compile(r"^[A-Za-z]{0,10}_?\d{2,}(\.|_|$)")
FILE_TOKENS = {}


def index_file_tokens(cands):
    """A record may list a directory's members by bare file name (V3_PROBE_SETS.files =
    ['nsm_answer_mids.parquet', ...]).  For every directory candidate collect its distinctive
    member names (>= 10 chars, '_' or '.', not a numbered part like part_00012.parquet) and
    drop any name that occurs under more than one candidate, so a shared name never keeps
    the wrong directory."""
    per = {}
    for _b, c in cands:
        ap = os.path.join(ROOT, c)
        if not os.path.isdir(ap):
            continue
        names = set()
        for dp, _dn, fn in os.walk(ap):
            for f in fn:
                if len(f) >= 10 and ("_" in f or "." in f) and not NUMBERED_PART.match(f) \
                        and not f.endswith((".done", ".tmp", ".lock")):
                    names.add(f)
            if len(names) > 400:
                break
        per[c] = names
    count = {}
    for names in per.values():
        for n in names:
            count[n] = count.get(n, 0) + 1
    FILE_TOKENS.clear()
    for c, names in per.items():
        FILE_TOKENS[c] = {n for n in names if count[n] == 1}


CHUNK_RE = re.compile(r"[A-Za-z0-9_.\-/]+")


def name_hits(text, tokens):
    """Every alt token (see alt_tokens) that occurs in text with identifier boundaries.

    Set membership over path-like chunks instead of one giant alternation regex: a chunk is a
    maximal run of [A-Za-z0-9_.-/]; each '/'-suffix of it is a candidate ('a/b/c.parquet' ->
    'a/b/c.parquet', 'b/c.parquet', 'c.parquet') and each directory segment followed by '/'
    is a candidate for the trailing-slash tokens ('mb_hits/')."""
    hits = set()
    for ch in CHUNK_RE.findall(text):
        if "_" not in ch and "." not in ch and "/" not in ch:
            continue
        segs = ch.strip("/").split("/")
        for i in range(len(segs)):
            suf = "/".join(segs[i:])
            if suf in tokens:
                hits.add(suf)
            if i < len(segs) - 1 and segs[i] + "/" in tokens:
                hits.add(segs[i] + "/")
    return hits


def harvest(files, tokens=None):
    """path-string -> set(files) for every path-like token in the given files; the second
    index maps every alt token (see alt_tokens) to the files that mention it."""
    idx = {}
    alt = {}
    n = 0
    for p in files:
        n += 1
        try:
            t = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        rel = norm(os.path.relpath(p, ROOT))
        t = BACKSLASHES.sub("/", t)  # Windows path forms (mb_hits\part_00000.parquet) count too
        for m in set(PATH_RE.findall(t)):
            idx.setdefault(m, set()).add(rel)
        if tokens:
            for m in name_hits(t, tokens):
                alt.setdefault(m, set()).add(rel)
    return (idx, alt), n


def name_refs(cand, alt_idx, exclude_self=True):
    out = set()
    for tok in alt_tokens(cand):
        for f in alt_idx.get(tok, ()):
            if exclude_self and (f == cand or f.startswith(cand + "/")):
                continue
            out.add(f)
    return sorted(out)


def refs_for(cand, idx, exclude_self=True):
    out = set()
    for tok, files in idx.items():
        if tok == cand or tok.startswith(cand + "/") or (cand.endswith(".npz") and tok == cand):
            for f in files:
                if exclude_self and (f == cand or f.startswith(cand + "/")):
                    continue
                out.add(f)
    return sorted(out)


def main():
    t0 = time.time()
    out_path = OUT
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            out_path = a.split("=", 1)[1]

    cands = expand()
    print("candidates:", len(cands))

    # sources
    rec_files = list(iter_files(FC, (".json", ".md"), 50_000_000)) + \
        list(iter_files("data/_family_v1", (".json", ".md"), 50_000_000))
    rec_files = [p for p in rec_files if not norm(p).endswith("CLEANUP_REFERENCE_CROSSCHECK.json")]
    code_live = list(iter_files("src", (".py", ".yaml", ".yml", ".toml", ".json", ".cfg"), 20_000_000)) + \
        list(iter_files("configs", (".py", ".yaml", ".yml", ".toml", ".json", ".cfg"), 20_000_000)) + \
        list(iter_files("baselines", (".py", ".yaml", ".yml", ".toml", ".json", ".cfg", ".md"), 20_000_000)) + \
        [os.path.join(ROOT, "experiments.py")]
    code_live = [p for p in code_live if os.path.exists(p) and "cleanup_reference_crosscheck" not in p
                 and "cleanup_pass_3" not in p]
    code_hist = list(iter_files("scratchpad", (".py", ".yaml", ".yml", ".toml", ".json", ".cfg", ".sh"), 20_000_000))
    res_files = list(iter_files("results", (".json", ".md", ".csv", ".txt"), 20_000_000)) + \
        list(iter_files("archive", (".json", ".md", ".csv", ".txt"), 20_000_000)) + \
        list(iter_files("docs", (".md", ".json"), 20_000_000))
    print("scanning: records %d  live-code %d  scratchpad-code %d  results %d"
          % (len(rec_files), len(code_live), len(code_hist), len(res_files)))
    index_file_tokens(cands)
    toks = set(t for _b, c in cands for t in alt_tokens(c))
    print("name tokens:", len(toks))
    (idx_rec, alt_rec), _ = harvest(rec_files, toks)
    (idx_live, alt_live), _ = harvest(code_live, toks)
    (idx_hist, alt_hist), _ = harvest(code_hist, toks)
    (idx_res, alt_res), _ = harvest(res_files, toks)

    pi = json.load(io.open(os.path.join(ROOT, FC, "POINTER_INDEX.json"), encoding="utf-8"))
    store_paths = set()
    for sec in ("datasets", "queries"):
        for ds, v in pi[sec].items():
            for ch in ("dense", "splade"):
                for s in v.get(ch, {}).get("stores", []):
                    store_paths.add(norm(s["path"]))
    rc = json.load(io.open(os.path.join(ROOT, FC, "RETRIEVAL_CACHE.json"), encoding="utf-8"))
    rc_paths = set(m for m in PATH_RE.findall(json.dumps(rc))
                   if m.count("/") >= 3 and m.startswith("data/final_canonical/"))

    # substrate tokens for the legacy loaders
    live_blob = "\n".join(io.open(p, encoding="utf-8", errors="replace").read() for p in code_live)

    rows = []
    totals = {}
    for bucket, c in cands:
        exists = os.path.exists(os.path.join(ROOT, c))
        b, n = tree_size(c) if exists else (0, 0)
        n_rec, n_live, n_hist, n_res = (name_refs(c, alt_rec), name_refs(c, alt_live),
                                        name_refs(c, alt_hist), name_refs(c, alt_res))
        by_name = sorted(set(n_rec) | set(n_live) | set(n_hist) | set(n_res))
        r_rec_all = sorted(set(refs_for(c, idx_rec)) | set(n_rec))
        r_rec = [f for f in r_rec_all if os.path.basename(f) not in LABELLING_RECORDS]
        labelled_by = [f for f in r_rec_all if os.path.basename(f) in LABELLING_RECORDS]
        r_live = sorted(set(refs_for(c, idx_live)) | set(n_live))
        r_hist = sorted(set(refs_for(c, idx_hist)) | set(n_hist))
        r_res = sorted(set(refs_for(c, idx_res)) | set(n_res))
        resolves = sorted(s for s in store_paths | rc_paths
                          if "/" in s and (s == c or s.startswith(c + "/") or c.startswith(s + "/")))
        token_hits = None
        if c.startswith("data/ukb_storage/") or c.startswith("data/l2_corpus/"):
            tok = c.split("/")[2]
            token_hits = len(re.findall(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(tok), live_blob))
        keep_reasons = []
        if r_rec:
            keep_reasons.append("cited by %d package record(s)" % len(r_rec))
        if r_live:
            keep_reasons.append("referenced by %d live code file(s) under src/configs/baselines" % len(r_live))
        if r_res:
            keep_reasons.append("referenced by %d results/archive/docs file(s)" % len(r_res))
        if resolves:
            keep_reasons.append("a POINTER_INDEX store or retrieval cache resolves here")
        if token_hits:
            keep_reasons.append("substrate token '%s' appears %d times in live code" % (c.split("/")[2], token_hits))
        if bucket == "E_PROVENANCE_KEEP_BY_RULE":
            keep_reasons.append("PROVENANCE rule: source text/TSV of live vectors or graphs, or upstream source")
        verdict = "KEEP" if keep_reasons else ("ABSENT" if not exists else "DELETE")
        rows.append({"bucket": bucket, "path": c, "exists": exists, "bytes": b, "files": n,
                     "verdict": verdict, "keep_reasons": keep_reasons,
                     "cited_by_records": r_rec[:25], "labelled_by": labelled_by,
                     "referenced_by_live_code": r_live[:25],
                     "referenced_by_scratchpad_code": r_hist[:25],
                     "referenced_by_results": r_res[:25], "resolves": resolves[:10],
                     "cited_by_name": by_name[:25], "name_tokens": sorted(alt_tokens(c)),
                     "substrate_token_hits_in_live_code": token_hits})
        t = totals.setdefault(bucket, {"DELETE": [0, 0], "KEEP": [0, 0], "ABSENT": [0, 0]})
        t[verdict][0] += b
        t[verdict][1] += 1

    rec = {
        "RECORD": "CLEANUP_REFERENCE_CROSSCHECK",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "READ_ONLY": True,
        "_what": ("per-path reference check for every cleanup candidate outside what "
                  "cleanup_plan.py already classifies; cleanup_pass_3.py deletes only paths "
                  "marked DELETE here"),
        "sources_scanned": {"records": len(rec_files), "live_code": len(code_live),
                            "scratchpad_code": len(code_hist), "results_archive_docs": len(res_files),
                            "pointer_store_paths": len(store_paths), "retrieval_cache_paths": len(rc_paths)},
        "verdict_rule": ("DELETE iff no record citation (by path OR by freebase_v3-relative "
                         "path OR by distinctive basename), no live-code reference, no results "
                         "reference, nothing resolves into it, and (ukb_storage/l2_corpus) the "
                         "substrate token is absent from live code; scratchpad-only references "
                         "and citations from the labelling records are listed, not counted"),
        "labelling_records_not_counted": sorted(LABELLING_RECORDS),
        "results_source_excludes": "archive/recovered_chats (transcripts, not provenance)",
        "protected_never_candidates": PROTECTED,
        "totals_by_bucket": {b: {k: {"bytes": v[0], "n": v[1]} for k, v in t.items()}
                             for b, t in totals.items()},
        "candidates": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with io.open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    for b, t in totals.items():
        print("%-32s DELETE %7.2f GB (%3d)   KEEP %7.2f GB (%3d)   ABSENT %d"
              % (b, t["DELETE"][0] / 1e9, t["DELETE"][1], t["KEEP"][0] / 1e9, t["KEEP"][1], t["ABSENT"][1]))
    print("wrote", norm(os.path.relpath(out_path, ROOT)), "in %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
