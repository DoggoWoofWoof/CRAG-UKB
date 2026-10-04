"""L2 CLEANUP (user 2026-09-30: "anything related to l2 is cleanup ... keep only the results and docs for l2 nothing more").

Deletes the L2 models, feature caches and training substrate and keeps the L2 results and docs.  DRY RUN by default: it lists every file
and byte and writes nothing.  Nothing is recoverable afterwards (the files are untracked in git and no archive is made), so read the plan first.

KEPT (never touched): every record file (.json .md .csv .txt .log .tsv .py .yaml .yml) anywhere in scope, every file < 1 MiB under
results/L2 (per-query result arrays, small stats), the per-query prediction dumps results/L2/*_preds.pt (results, 17 MB in all), every
_summary.json under data/l2_corpus, and everything outside the scope below (L1 / L3 / DATA_FREEZE / final_canonical / l1_canonical /
G2_L1_PARTITION_SEARCH / ablation_qwen / _l1* scratch / ukb_storage/<ds>/gte_qwen ...).
DELETED (scope):
  results/L2             binary files >= 1 MiB: relsig_feats_*.pt, signals_*.npz, _ctrl/*.joblib|npz feature caches and trained rerankers, ckpts
  data/l2_corpus         the C11a training substrate (candidate rank / score arrays), all but _summary.json
  scratchpad/_ctrl_cache the controller feature cache
  scratchpad/            _relcache_master_*.pkl, _reledges_*.pkl, _relscore_*.npy, _qwen_sent_emb.npy (L2 relation caches, read only by l2_relation*.py)
OPT-IN (--with-head-cache): data/ukb_storage/_head_cache/*.pt (trained universal-head / dense-adapter caches; written by src/experiments
  l1_universal_head.py, dense_adapter.py and e2e_pipeline.py, read by the l2-seed / l2-learned-fusion / e2e-ner tasks).

Usage: python scratchpad/l2_cleanup.py                     -> dry run (lists the plan, deletes nothing)
       python scratchpad/l2_cleanup.py --apply             -> deletes the plan, writes results/DATA_CLEANUP/L2_DELETION_MANIFEST__2026-09-30.json
       add --with-head-cache to include the opt-in group
"""
import json
import os
import sys
import time

REPO = "C:/Users/Swastik/Desktop/CRAG"
MANIFEST = REPO + "/results/DATA_CLEANUP/L2_DELETION_MANIFEST__2026-09-30.json"
MIB = 1 << 20
RECORD_EXT = {".json", ".md", ".csv", ".txt", ".log", ".tsv", ".py", ".yaml", ".yml"}
PROTECT = ("data/final_canonical", "data/l1_canonical", "data/l1_lowmem", "data/original", "data/processed", "data/_family_v1",
           "results/L1", "results/L3", "results/GENERALIZATION", "results/DATA_", "src/", ".git/", "archive/")


def lp(p):
    p = os.path.abspath(p).replace("/", "\\")
    return p if p.startswith("\\\\?\\") else "\\\\?\\" + p


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def walk(root):
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(lp(d)) as it:
                for e in it:
                    q = d.rstrip("/") + "/" + e.name
                    if e.is_symlink():
                        continue
                    if e.is_dir(follow_symlinks=False):
                        stack.append(q)
                    elif e.is_file(follow_symlinks=False):
                        st = e.stat(follow_symlinks=False)
                        yield q, st.st_size, st.st_mtime_ns
        except OSError:
            pass


def plan(with_head):
    out = []                                   # (group, path, size, mtime_ns)
    for p, sz, mt in walk(REPO + "/results/L2"):
        r = rel(p)
        ext = os.path.splitext(p)[1].lower()
        if ext in RECORD_EXT or sz < MIB or (r.startswith("results/L2/") and r.count("/") == 2 and r.endswith("_preds.pt")):
            continue
        out.append(("results/L2", p, sz, mt))
    for p, sz, mt in walk(REPO + "/data/l2_corpus"):
        if os.path.basename(p) != "_summary.json":
            out.append(("data/l2_corpus", p, sz, mt))
    for p, sz, mt in walk(REPO + "/scratchpad/_ctrl_cache"):
        out.append(("scratchpad/_ctrl_cache", p, sz, mt))
    sp = REPO + "/scratchpad"
    for name in sorted(os.listdir(sp)):
        if (name.startswith("_relcache_master_") and name.endswith(".pkl")) or (name.startswith("_reledges_") and name.endswith(".pkl")) \
                or (name.startswith("_relscore_") and name.endswith(".npy")) or name == "_qwen_sent_emb.npy":
            q = sp + "/" + name
            if os.path.isfile(lp(q)):
                st = os.stat(lp(q))
                out.append(("scratchpad/rel_caches", q, st.st_size, st.st_mtime_ns))
    if with_head:
        for p, sz, mt in walk(REPO + "/data/ukb_storage/_head_cache"):
            out.append(("ukb_storage/_head_cache", p, sz, mt))
    for g, p, sz, mt in out:
        r = rel(p)
        assert not any(r.startswith(x) for x in PROTECT), "protected path in plan: " + r
    return out


def main():
    apply_ = "--apply" in sys.argv
    with_head = "--with-head-cache" in sys.argv
    pl = plan(with_head)
    by = {}
    for g, p, sz, mt in pl:
        by.setdefault(g, [0, 0])
        by[g][0] += 1
        by[g][1] += sz
    print("PLAN (%s):" % ("APPLY" if apply_ else "dry run - nothing is deleted"))
    for g, (n, b) in by.items():
        print("  %-26s %4d files %9.3f GB" % (g, n, b / 1e9))
    tot = sum(v[1] for v in by.values())
    print("  %-26s %4d files %9.3f GB" % ("TOTAL", len(pl), tot / 1e9))
    if not apply_:
        print("largest 15:")
        for g, p, sz, mt in sorted(pl, key=lambda x: -x[2])[:15]:
            print("  %8.1f MB  %s" % (sz / 1e6, rel(p)))
        print("re-run with --apply to delete.")
        return
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "with_head_cache": with_head,
           "files": [{"group": g, "path": rel(p), "bytes": sz, "mtime_ns": mt} for g, p, sz, mt in pl], "total_bytes": tot}
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:      # manifest first: a crash mid-delete still leaves the record
        json.dump(rec, f, indent=1, sort_keys=True)
    gone = 0
    freed = 0
    for g, p, sz, mt in pl:
        st = os.stat(lp(p))
        if st.st_size != sz or st.st_mtime_ns != mt:
            print("SKIP (changed since the plan): " + rel(p))
            continue
        try:
            os.remove(lp(p))
            gone += 1
            freed += sz
        except OSError as e:
            print("KEPT (cannot delete): %s: %s" % (rel(p), e))
    for root in ("data/l2_corpus", "scratchpad/_ctrl_cache", "data/ukb_storage/_head_cache"):
        for r_, dirs, fs in os.walk(lp(REPO + "/" + root), topdown=False):       # only directories the deletion emptied
            if not os.listdir(r_) and rel(r_[4:]) != root:
                os.rmdir(r_)
    print("DELETED %d of %d files, %.3f GB freed" % (gone, len(pl), freed / 1e9))


if __name__ == "__main__":
    main()
