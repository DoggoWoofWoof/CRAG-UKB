"""Host housekeeping -- delete OUR OWN directories on the lab host from an explicit allowlist (never wildcards, never anything outside the rx workspace, never through a junction/reparse point such as
data/final_canonical/freebase -> the project data store).  Default is a dry run; --apply deletes.

  python -u scratchpad/_host_rm.py <PHASE> [--apply]

Phases (each lists directories relative to the rx workspace):
  PHASE1   pure scratch: ooc/vc scratch, regression scratch, the encode smoke-test dir, the expired signed-link cache.  (NOT the Hotpot/2Wiki substrates: the host L1 text scorer + held-out eval read them.)
  PHASE2   work/L1_HOST/phg_data (Zoltan/Mt-KaHyPar per-rank inputs + run dirs of the completed L1_HOST lane; the small records were packed + fetched first, the maps are on the laptop)
  PHASE3   completed-experiment outputs; every file must be in the verified HF restore index (data/_cache/hfx_known.json pushed with --inputs), otherwise that directory is BLOCKED
"""
import os
import shutil
import stat
import sys
import time

WS = os.path.realpath(os.getcwd())
PHASES = {
    "PHASE1": [
        "data/ooc_tmp", "data/vc_tmp", "data/freebase_scale/enc_test",
        "work/FBX_SCALE/webqsp_LP_S_regr", "work/FBX_SCALE/webqsp_LDG_S_regr", "work/FBX_SCALE/webqsp_HASH_regr",
        "data/_cache/fbx_urls.json",              # NOT data/final_canonical/{2wiki,hotpotqa}: the host-side L1 text scorer + the held-out L1 eval still read them (user, 2026-10-03)
    ],
    "PHASE2": ["work/L1_HOST/phg_data"],
    # completed FREEBASE_SCALE experiments (ML / H2L / IVF arms: refuted or infeasible, cataloged in the laptop records + E-reports): their big outputs are verified on HF (Swastik9895/crag-host-backup,
    # restore index MANIFEST/RESTORE_INDEX.json, groups r_ml2 / r_h2l / r_ivf / recs_big).  results/FREEBASE_SCALE/parts_S (the S-arm reference maps) STAYS: the regression gate reads it.
    "PHASE3": ["results/FREEBASE_SCALE/ml2", "results/FREEBASE_SCALE/h2l", "results/FREEBASE_SCALE/ivf"],
    # 2026-10-04 (user: remove what is not required, periodically): HF-verified AND not read by any running/queued job -- the Freebase NER families (HF group fbs_ner; the encode job needs names + tree, NOT ner; restore
    # ~6 min on the host at 40 MB/s) and the CALIB bundle (HF group calib incl. ford.npy; the CALIB EVAL gates are finished).  Same HF-index check as PHASE3.
    "PHASE4": ["data/freebase_scale/ner", "work/FBX_CALIB/bundle_v1"],
}
SKIP = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--skip=")]      # directories pinned by the laptop (data/_cache/HOUSEKEEP_PINS.json): never deleted this run
REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def is_reparse(p):
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return bool(getattr(st, "st_file_attributes", 0) & REPARSE) or stat.S_ISLNK(st.st_mode)


def census(p):
    n = 0
    b = 0
    if os.path.isfile(p):
        return 1, os.path.getsize(p), []
    bad = []
    for dp, dn, fn in os.walk(p):
        for d in list(dn):
            if is_reparse(os.path.join(dp, d)):
                bad.append(os.path.join(dp, d))
                dn.remove(d)
        for f in fn:
            fp = os.path.join(dp, f)
            if is_reparse(fp):
                bad.append(fp)
                continue
            try:
                b += os.lstat(fp).st_size
                n += 1
            except OSError:
                pass
    return n, b, bad


def not_on_hf(p):
    """files under p whose (workspace-relative path | size) is not in the HF restore index (data/_cache/hfx_known.json, made by `_hfx.py KNOWN` on the laptop and pushed with --inputs)"""
    import json
    kp = os.path.join(WS, "data", "_cache", "hfx_known.json")
    known = set(json.load(open(kp)))
    miss = []
    for dp, dn, fn in os.walk(p):
        for f in fn:
            fp = os.path.join(dp, f)
            rel = os.path.relpath(fp, WS).replace("\\", "/")
            if "%s|%d" % (rel, os.lstat(fp).st_size) not in known:
                miss.append(rel)
    return miss


def main():
    for phase in [a for a in sys.argv[1:] if a.startswith("PHASE")]:
        one(phase, "--apply" in sys.argv)


def one(phase, apply):
    free0 = shutil.disk_usage(WS).free
    tot = 0
    for rel in PHASES[phase]:
        assert ".." not in rel and not rel.startswith(("/", "\\")) and ":" not in rel and "*" not in rel, rel
        if rel in SKIP:
            print("pinned  ", rel, "(--skip: a job needs it)", flush=True)
            continue
        p = os.path.join(WS, rel.replace("/", os.sep))
        rp = os.path.realpath(p)
        if not os.path.exists(p):
            print("absent  ", rel, flush=True)
            continue
        assert rp.startswith(WS + os.sep) and os.path.normcase(rp) == os.path.normcase(os.path.abspath(p)), "path resolves outside the workspace (junction?): %s -> %s" % (p, rp)
        assert not is_reparse(p), "top path is itself a link: %s" % p
        n, b, bad = census(p)
        print("%-8s %-52s %7d files %9.3f GB%s" % ("DELETE" if apply else "would", rel, n, b / 1e9, ("   [BLOCKED: %d reparse points inside, e.g. %s]" % (len(bad), bad[0])) if bad else ""), flush=True)
        if bad:
            continue
        if phase in ("PHASE3", "PHASE4"):
            miss = not_on_hf(p)
            if miss:
                print("BLOCKED  %s: %d files are not in the verified HF backup, e.g. %s" % (rel, len(miss), miss[0]), flush=True)
                continue
        tot += b
        if apply:
            if os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p)
            assert not os.path.exists(p), p
    free1 = shutil.disk_usage(WS).free
    print("%s: %.2f GB %s; drive free %.1f -> %.1f GB at %s" % (phase, tot / 1e9, "deleted" if apply else "would be deleted", free0 / 1e9, free1 / 1e9, time.strftime("%F %T")), flush=True)


main()
