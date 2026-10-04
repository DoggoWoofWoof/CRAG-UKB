"""Phase 4 -- QUERY-INDEPENDENCE HARD TEST + deterministic rebuild.

    python scratchpad/final_canonical_build/qi_test.py <ds> [--runs A,B,none]

Runs the dataset's builder three times into data/final_canonical/_work/qi/<ds>/{A,B,none}:
    A    --eval-subset legacy                     (the frozen legacy subset)
    B    --eval-subset random_dev:7 / random_held:7  (a different fixed subset)
    none --eval-subset none

Builder dispatch: build.py for metaqa/musique/squad, build_kb.py for 2wiki/hotpotqa/webqsp (see KB below).
COST NOTE: each run re-derives the corpus from the official sources -- that is what makes the test
constructive rather than bookkeeping. For hotpotqa one pass streams the 1.5 GB tarball (Phase-C measured
761 s for the read alone), so the full three-run test is a ~1 h job. --runs can drop B if the machine is
busy; dropping it must then be disclosed, since it removes the third distinct query subset.
and compares them with the FINAL build at data/final_canonical/<ds> (which used --eval-subset legacy, i.e. the
same command as A up to --out -> A vs FINAL is the deterministic-rebuild check).

Asserts across all four: identical node count, identical node-id SET (streamed, not just hashed), identical
ordered node-id hash, identical corpus content hash, identical nodes.jsonl sha256.
Writes data/final_canonical/<ds>/query_independence_test.json. Removes the bulky nodes.jsonl / queries of the
three test builds afterwards (reports kept).
"""
import sys, os, json, subprocess, hashlib, shutil, time

ROOT = "data/final_canonical"; WORK = f"{ROOT}/_work/qi"
BUILD = "scratchpad/final_canonical_build/build.py"
BUILD_KB = "scratchpad/final_canonical_build/build_kb.py"
KB = {"webqsp", "hotpotqa", "2wiki"}  # built by build_kb.py; their second subset flag is random_held
#   2wiki moved here on 2026-09-05: its canonical corpus is now the FULL para_with_hyperlink article universe
#   (~6M records), which only the streaming/external-sort builder can produce.  random_held:7 draws from its
#   official DEV split (SRC['2wiki']['held_out_split']), i.e. exactly what random_dev:7 meant under build.py.


def ids_of(path):
    s = set(); n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s.add(json.loads(line)["node_id"]); n += 1
    return s, n


def main(ds, only=None):
    t0 = time.time()
    builder = BUILD_KB if ds in KB else BUILD
    runs = {"A": "legacy", "B": ("random_held:7" if ds in KB else "random_dev:7"), "none": "none"}
    if only:
        runs = {k: v for k, v in runs.items() if k in only}
        assert "A" in runs and "none" in runs, "--runs must keep at least A (determinism) and none (independence)"
    cmds = {}
    env = dict(os.environ, PYTHONUTF8="1")
    for tag, sub in runs.items():
        out = f"{WORK}/{ds}/{tag}"
        cmd = [sys.executable, builder, ds, "--eval-subset", sub, "--out", out]
        cmds[tag] = " ".join(cmd)
        print(f"[qi {ds}] run {tag}: {cmds[tag]}", flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f"build {tag} failed")
    final = f"{ROOT}/{ds}"
    dirs = {"FINAL": final, **{t: f"{WORK}/{ds}/{t}" for t in runs}}
    rep = {}
    for tag, d in dirs.items():
        ir = json.load(open(f"{d}/integrity_report.json", encoding="utf-8"))
        bi = json.load(open(f"{d}/build_info.json", encoding="utf-8"))
        rep[tag] = {"dir": d, "eval_subset": bi["eval_subset_arg"], "canonical_node_count": ir["canonical_node_count"],
                    "CORPUS_HASH": ir["CORPUS_HASH"], "NODE_ORDER_HASH": ir["NODE_ORDER_HASH"], "nodes_jsonl_sha256": ir["nodes_jsonl_sha256"],
                    "eval_file": (ir.get("eval_subset") or {}).get("file"), "eval_n": (ir.get("eval_subset") or {}).get("n"),
                    "builder_sha256": bi["builder_sha256"], "command": bi["command"]}
    # direct node-id SET comparison (A vs B vs none vs FINAL)
    base_ids, base_n = ids_of(f"{final}/nodes.jsonl")
    set_equal = {}
    for tag in runs:
        s, n = ids_of(f"{dirs[tag]}/nodes.jsonl")
        set_equal[tag] = {"same_count": n == base_n, "same_id_set": s == base_ids,
                          "only_in_final": len(base_ids - s), "only_in_run": len(s - base_ids)}
    keys = ["canonical_node_count", "CORPUS_HASH", "NODE_ORDER_HASH", "nodes_jsonl_sha256"]
    all_equal = all(rep[t][k] == rep["FINAL"][k] for t in runs for k in keys) and all(v["same_id_set"] for v in set_equal.values())
    # the eval subsets must actually differ across runs (otherwise the test is vacuous).
    # 'none' produces no eval file at all, so A vs none is already a real difference; B adds a third subset.
    a_ids = {json.loads(l)["query_id"] for l in open(rep["A"]["eval_file"], encoding="utf-8")}
    if "B" in runs:
        b_ids = {json.loads(l)["query_id"] for l in open(rep["B"]["eval_file"], encoding="utf-8")}
    else:
        b_ids = set()
    subsets_differ = a_ids != b_ids
    deterministic = all(rep["A"][k] == rep["FINAL"][k] for k in keys) and rep["A"]["builder_sha256"] == rep["FINAL"]["builder_sha256"]
    result = {"dataset": ds, "runs": rep, "builder": builder, "runs_executed": sorted(runs),
              "node_id_set_equality_vs_FINAL": set_equal,
              "eval_subsets_A_vs_B": {"n_A": len(a_ids), "n_B": len(b_ids), "overlap": len(a_ids & b_ids), "differ": subsets_differ,
                                      "B_executed": "B" in runs},
              "CORPUS_QUERY_INDEPENDENT": bool(all_equal and subsets_differ),
              "DETERMINISTIC_REBUILD": bool(deterministic),
              "commands": cmds, "seconds": round(time.time() - t0, 1)}
    json.dump(result, open(f"{final}/query_independence_test.json", "w", encoding="utf-8"), indent=2)
    # cleanup bulky test artefacts (keep reports + eval files)
    for tag in runs:
        d = dirs[tag]
        for p in (f"{d}/nodes.jsonl",):
            if os.path.exists(p): os.remove(p)
        if os.path.isdir(f"{d}/queries"): shutil.rmtree(f"{d}/queries")
    line = f"[qi {ds}] CORPUS_QUERY_INDEPENDENT={result['CORPUS_QUERY_INDEPENDENT']} DETERMINISTIC_REBUILD={result['DETERMINISTIC_REBUILD']} ({result['seconds']}s)"
    print(line, flush=True)
    with open(f"{ROOT}/_build.log", "a", encoding="utf-8") as f: f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + line + "\n")
    if not result["CORPUS_QUERY_INDEPENDENT"]:
        raise SystemExit("QUERY INDEPENDENCE FAILED -- see query_independence_test.json")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    sel = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--runs=")), None)
    main(pos[0], set(sel.split(",")) if sel else None)
