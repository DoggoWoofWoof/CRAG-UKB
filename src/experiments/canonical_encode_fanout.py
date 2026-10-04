"""C5 fan-out scheduler — deterministic multi-account, multi-SLOT encoding.

Job unit = (dataset, kind, model, shard_id), fixed SHARD_SIZE so ids are stable and every shard has
exactly one active owner (account, slot). Completed shards are never rerun. Credentials are env-only, so
each worker is a separate OS process permanently bound to one --account; an account runs up to
SLOTS_PER_ACCT (default 3, max 4) concurrent Modal containers. A worker encodes a small BLOCK of shards
via `experiments.py run canonical-encode --backend modal --account <i> -- ... --shard-ids <block>`.

Global work queue (dynamic load balance): when any slot frees, it pulls the next eligible shard-block —
no static dataset→account assignment. Priority: short jobs (queries, small-dataset doc shards) first to
validate throughput, then the long poles (hotpot/2wiki/webqsp docs). Account hitting spend-limit → no new
scheduling to its slots, running jobs allowed to finish, its unstarted shards requeue elsewhere. A
deterministic model/data error FAILS the shard (no rotation — rotation must not hide bugs). Coordinator
(this process) alone writes manifest.json/index.json via canonical_encode.finalize. Live status + ETA.
"""
import os, sys, json, time, logging, argparse, subprocess, collections, itertools
from src.experiments import credentials, canonical_encode as CE

log = logging.getLogger("c5.fanout")
ROOT = os.path.abspath(".")
LOGDIR = "scratchpad/c5_workers"; os.makedirs(LOGDIR, exist_ok=True)
DENSE, SPLADE = "dense", "splade"
BLOCK = {DENSE: 2, SPLADE: 3}                    # shards per worker (amortize container cold-start)
BATCH = {DENSE: 32, SPLADE: 64}
SPEND = ("spend limit", "exceeded its", "resourceexhausted", "quota", "no capacity")
_wid = itertools.count()


def default_groups():
    """Short/validation jobs first (queries + small docs), then the long document poles."""
    small_docs = ["squad", "metaqa", "musique"]
    big_docs   = ["2wiki", "webqsp", "hotpotqa", "2wiki_universe"]  # universe = full ~5.99M reusable substrate (docs only, no queries)
    query_ds   = ["webqsp", "musique", "squad", "metaqa_1hop", "metaqa_2hop", "metaqa_3hop",
                  "2wiki", "hotpotqa"]
    g = []
    for m in (DENSE, SPLADE):
        for ds in query_ds:  g.append((ds, "queries", m))      # wave 1a: tiny queries
    for m in (DENSE, SPLADE):
        for ds in small_docs: g.append((ds, "docs", m))        # wave 1b: small docs
    for m in (DENSE, SPLADE):
        for ds in big_docs:  g.append((ds, "docs", m))         # wave 2: long poles
    return g


def _rows(ds, kind, s, ntot=None):
    ntot = ntot if ntot is not None else CE.n_items(ds, kind); ss = CE.SHARD_SIZE
    return min((s + 1) * ss, ntot) - s * ss

def enumerate_missing(group):
    ds, kind, model = group
    ns = CE.pre_shard(ds, kind); ntot = CE.n_items(ds, kind)
    missing = [s for s in range(ns) if not CE.shard_complete(ds, kind, model, s, expect_rows=_rows(ds, kind, s, ntot))]
    return ns, ntot, missing

def build_queue(groups):
    q = collections.deque(); plan = []
    for g in groups:
        ns, ntot, missing = enumerate_missing(g)
        plan.append((g, ns, len(missing), ntot))
        for i in range(0, len(missing), BLOCK[g[2]]):
            q.append((g, missing[i:i + BLOCK[g[2]]]))
    return q, plan

def classify_fail(logpath):
    try: txt = open(logpath, encoding="utf-8", errors="replace").read().lower()
    except Exception: return "TEMP_FAILED"
    if any(s in txt for s in SPEND): return "SPEND_LIMIT"
    if "outofmemory" in txt or "cuda out of memory" in txt: return "OOM"
    if any(s in txt for s in ("unauthor", "invalid token", "token not found",
                              "autherror", "auth error", "authentication")): return "AUTH_FAILED"
    return "TEMP_FAILED"

def launch_worker(acct, group, shards):
    ds, kind, model = group
    ids = ",".join(str(s) for s in shards)
    logp = os.path.join(LOGDIR, f"{ds}_{kind}_{model}_a{acct}_{shards[0]:05d}.log")
    argv = [sys.executable, "experiments.py", "run", "canonical-encode",
            "--backend", "modal", "--account", str(acct), "--",
            "--dataset", ds, "--kind", kind, "--model", model,
            "--shard-ids", ids, "--batch", str(BATCH[model])]
    fh = open(logp, "w", encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    p = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    return {"id": next(_wid), "proc": p, "log": logp, "fh": fh, "group": group,
            "shards": shards, "acct": acct, "t0": time.time(),
            "rows": sum(_rows(ds, kind, s) for s in shards)}

def validate_block(group, shards):
    ds, kind, model = group
    done = [s for s in shards if CE.shard_complete(ds, kind, model, s, expect_rows=_rows(ds, kind, s))]
    bad = [s for s in shards if s not in done]
    return done, bad


def run(groups, only=None, poll=20, slots=3, max_slots=4):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if only:
        want = set(tuple(o.split(":")) for o in only); groups = [g for g in groups if g in want]
    pool = credentials.load_pool("modal")
    accounts = {i: "ACTIVE" for i in range(len(pool))}
    # seed disabled from a prior health probe if present
    hp = "scratchpad/modal_health.json"
    if os.path.exists(hp):
        for k, v in json.load(open(hp)).items():
            if v.get("status") != "ACTIVE": accounts[int(k)] = v["status"]
    queue, plan = build_queue(groups)
    total_shards = sum(len(sh) for _, sh in queue)
    running = {}                                            # wid -> worker
    slots_used = collections.Counter()                     # acct -> busy slots
    done_rows = {DENSE: 0, SPLADE: 0}; done_time = {DENSE: 0.0, SPLADE: 0.0}
    completed = 0; failed = 0; t0 = time.time(); bumped = False
    attempts = collections.Counter()                       # (group,shard) -> tries; cap requeues (no infinite loop on a code bug)
    dead_shards = set(); MAX_TRY = 3
    log.info("[c5] accounts=%d healthy=%d slots/acct=%d total_slots=%d queued_shards=%d",
             len(pool), sum(v=="ACTIVE" for v in accounts.values()), slots,
             sum(v=="ACTIVE" for v in accounts.values())*slots, total_shards)

    while queue or running:
        # ---- fill free slots on healthy accounts ----
        for acct in list(accounts):
            if accounts[acct] != "ACTIVE": continue
            while slots_used[acct] < slots and queue:
                g, shards = queue.popleft()
                w = launch_worker(acct, g, shards); running[w["id"]] = w; slots_used[acct] += 1
                log.info("[c5] launch a%d slot%d  %s/%s/%s shards=%s",
                         acct, slots_used[acct], g[0], g[1], g[2], shards)
        # ---- poll ----
        for wid in list(running):
            w = running[wid]; rc = w["proc"].poll()
            if rc is None: continue
            w["fh"].flush(); w["fh"].close(); slots_used[w["acct"]] -= 1
            done, bad = validate_block(w["group"], w["shards"]); dt = time.time() - w["t0"]
            m = w["group"][2]
            if done:
                done_rows[m] += sum(_rows(w["group"][0], w["group"][1], s) for s in done); done_time[m] += dt
            completed += len(done)
            if rc == 0 and not bad:
                log.info("[c5] OK a%d %s %s (%.0fs, %d rows)", w["acct"], "/".join(w["group"]), w["shards"], dt, w["rows"])
            else:
                cls = classify_fail(w["log"]) if rc != 0 else "TEMP_FAILED"
                log.warning("[c5] FAIL a%d %s rc=%s cls=%s done=%s bad=%s", w["acct"], "/".join(w["group"]), rc, cls, done, bad)
                if cls in ("SPEND_LIMIT", "AUTH_FAILED"): accounts[w["acct"]] = cls
                # requeue unfinished shards, but cap retries so a deterministic code/data bug can't loop forever
                requeue = []
                acct_fault = cls in ("SPEND_LIMIT", "AUTH_FAILED")   # account-level failure, not the shard's fault
                for s in bad:
                    key = (w["group"], s); attempts[key] += 1
                    if acct_fault or attempts[key] < MAX_TRY:   # dead-account requeues don't count against the shard
                        if acct_fault: attempts[key] -= 1
                        requeue.append(s)
                    else:
                        dead_shards.add(key); failed += 1
                        log.error("[c5] shard PERMANENTLY FAILED after %d tries: %s shard %d — investigate (not requeued)",
                                  attempts[key], "/".join(w["group"]), s)
                if requeue: queue.append((w["group"], requeue))
            del running[wid]
            if not any(v == "ACTIVE" for v in accounts.values()):
                log.error("[c5] no healthy accounts remain; %d shards queued, %d running", len(queue), len(running))
                _drain(running); _finalize_all(groups); return
        # ---- autotune: raise to 4 slots if throughput is healthy & stable ----
        if not bumped and slots < max_slots and completed >= 8 and failed == 0 \
           and all(v != "OOM" for v in accounts.values()):
            slots += 1; bumped = True
            log.info("[c5] autotune: raising slots/acct -> %d (healthy throughput, no failures)", slots)
        _status(plan, accounts, slots_used, queue, running, completed, failed, total_shards, done_rows, done_time, t0, slots)
        if queue or running: time.sleep(poll)
    log.info("[c5] queue drained; finalizing."); _finalize_all(groups)
    log.info("[c5] DONE completed=%d failed=%d elapsed=%.0fm", completed, failed, (time.time()-t0)/60)


def _status(plan, accounts, slots_used, queue, running, completed, failed, total, done_rows, done_time, t0, slots):
    healthy = [i for i, s in accounts.items() if s == "ACTIVE"]
    disabled = {i: s for i, s in accounts.items() if s != "ACTIVE"}
    dr = done_rows[DENSE] / done_time[DENSE] if done_time[DENSE] > 0 else 0
    sr = done_rows[SPLADE] / done_time[SPLADE] if done_time[SPLADE] > 0 else 0
    # ETA from remaining rows / observed aggregate rate (per-slot rate * concurrency)
    conc = max(1, len(running))
    agg = (dr + sr) * conc if (dr + sr) else 0
    rem_rows = 0
    for (g, ns, _m, ntot) in plan:
        for s in range(ns):
            if not CE.shard_complete(g[0], g[1], g[2], s): rem_rows += _rows(g[0], g[1], s, ntot)
    eta = rem_rows / agg if agg else float("inf")
    log.info("── C5 ── healthy %d/%d  slots/acct=%d  running=%d queued=%d complete=%d/%d failed=%d  "
             "dense=%.0f d/s splade=%.0f d/s  ETA=%s  disabled=%s",
             len(healthy), len(accounts), slots, len(running), len(queue), completed, total, failed,
             dr, sr, ("%.0fm" % (eta/60) if eta != float("inf") else "?"), disabled)
    busy = "  ".join(f"a{i}:{slots_used[i]}/{slots}" for i in healthy)
    if busy: log.info("    slots: %s", busy)
    for (g, ns, _m, _n) in plan:
        d = sum(1 for s in range(ns) if CE.shard_complete(g[0], g[1], g[2], s))
        if d < ns: log.info("    %-12s %-7s %-6s %4d/%-4d", g[0], g[1], g[2], d, ns)


def _drain(running):
    for w in running.values():
        try: w["proc"].terminate()
        except Exception: pass

def _finalize_all(groups):
    for g in groups:
        try:
            complete, present, ns = CE.finalize(*g)
            if present: log.info("[c5] finalize %s: %d/%d complete=%s", "/".join(g), len(present), ns, complete)
        except Exception as e:
            log.warning("[c5] finalize %s failed: %s", "/".join(g), e)


def main(argv=None):
    p = argparse.ArgumentParser(prog="canonical-encode-fanout")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--only", nargs="+", help="restrict to groups 'ds:kind:model'")
    p.add_argument("--poll", type=int, default=20)
    p.add_argument("--slots", type=int, default=3, help="concurrent Modal containers per account")
    p.add_argument("--max-slots", type=int, default=4)
    a = p.parse_args(argv)
    groups = default_groups()
    if a.only:
        want = set(tuple(o.split(":")) for o in a.only); groups = [g for g in groups if g in want]
    if a.plan:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _, plan = build_queue(groups)
        for (g, ns, miss, ntot) in plan:
            print(f"{g[0]:12s} {g[1]:7s} {g[2]:6s} shards={ns:4d} missing={miss:4d} items={ntot}")
        return
    run(groups, only=a.only, poll=a.poll, slots=a.slots, max_slots=a.max_slots)


if __name__ == "__main__":
    main()
