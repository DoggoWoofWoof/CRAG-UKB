"""Re-arm the WebQSP PHG lane under the tenth ruling (2026-09-15): supersede the 2026-09-14 WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED
run record (results/L1_LOWMEM/PHG_WEBQSP_RUNS.json -> _history/, sha-named, via the frozen phg_repair.supersede), move the exhausted
wait state to logs/_history/, record the host at re-arm (the frozen program's own preflight_sample) and write PHG_WEBQSP_REARM.json.
Then `phg_webqsp.py arm` starts a fresh 24 h wait budget and _phg_webqsp_watch.py runs the preflight-gated chain in the background.

Nothing frozen is edited: phg_webqsp.py and its modules are imported unchanged; the foreign process is never signalled; nothing is
launched here (the launch stays behind the frozen program's own preflight gate)."""
import io
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
from src.l1_lowmem.common import OUT, log, sha_file, rj, wj, pin  # noqa: E402
from src.l1_lowmem import phg_repair as PR  # noqa: E402
from src.l1_lowmem import phg_webqsp as W  # noqa: E402

RULING_TENTH = ("In parallel/requisite infrastructure, get WebQSP PHG running now that the foreign process is gone. "
                "3. Run frozen H2_PATCH1 -> WebQSP once, with no changes. "
                "4. Only after that open WIDE_H2_PATCH as a separate arm, with beam width and patch compression jointly specified.")
STANDING = W.PREFLIGHT["user_requirement_verbatim"]

rec_p = os.path.join(OUT, "PHG_WEBQSP_REARM.json")
if os.path.exists(rec_p):
    raise SystemExit("re-arm record exists: %s" % rec_p)
if os.path.exists(os.path.join(OUT, W.PREREG_W)) or os.path.exists(os.path.join(OUT, W.INPUTS_W)) or os.path.isdir(os.path.join(W.ds_dir(W.DS), "phg")):
    raise SystemExit("unexpected lane state (inputs / prereg / phg dir present)")

old_runs = rj(os.path.join(OUT, W.RUNS_W))
assert old_runs and old_runs["DECISION"] == "WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED", old_runs and old_runs.get("DECISION")
old_wait = rj(W.WAIT_STATE)
assert old_wait and time.time() > float(old_wait["deadline_epoch"]), "the previous wait budget has not elapsed"
old_state = rj(W.PREFLIGHT_STATE)

# host at re-arm: the frozen program's own sample (read-only; typeperf + wsl free); appended to the preflight log like every sample
host = W.preflight_sample(None, purpose="rearm")
W.append_log(dict(host, rearm=True))
log("  host at re-arm: %s" % W.one_line(host))
same_job = [p for p in host["foreign_processes"] if p["pid"] != W.FOREIGN_PID]

# supersede the NOT_RUN run record (frozen record -> _history/<stem>.<sha8>.json, unchanged bytes) and retire the exhausted wait state
moved_runs = PR.supersede(W.RUNS_W)
hist_logs = os.path.join(W.LOGDIR, "_history")
os.makedirs(hist_logs, exist_ok=True)
sha_w = sha_file(W.WAIT_STATE)
dst_w = os.path.join(hist_logs, "phg_webqsp_wait_state.%s.json" % sha_w[:8])
assert not os.path.exists(dst_w)
shutil.move(W.WAIT_STATE, dst_w)
assert sha_file(dst_w) == sha_w
moved_wait = {"path": os.path.relpath(dst_w, REPO).replace("\\", "/"), "sha256": sha_w, "bytes": os.path.getsize(dst_w)}

rec = {"RECORD": "PHG_WEBQSP_REARM", "utc": W.utc(), "classification": W.CLASS, "dataset": W.DS, "lane": W.LANE, "tag": W.TAG_W,
       "authorization": {"lane_ruling": pin(os.path.join(OUT, W.RULING_W)), "tenth_ruling_2026-09-15_verbatim": RULING_TENTH,
                         "standing_hygiene_requirement_verbatim": STANDING},
       "previous_outcome": {"run_record_superseded_to_history": moved_runs, "decision": old_runs["DECISION"], "utc": old_runs["utc"],
                            "preflight_history": old_runs["preflight"]["history_summary"], "wait_state_retired_to_logs_history": moved_wait,
                            "wait_state": {k: old_wait[k] for k in ("armed_utc", "deadline_utc", "wait_budget_hours")},
                            "report_left_in_place": {fn: pin(os.path.join(OUT, fn)) for fn in (W.REPORT_W + ".json", W.REPORT_W + ".md")},
                            "report_note": "PHG_WEBQSP_REPORT.{json,md} describe the 2026-09-14 NOT_RUN outcome; `phg_webqsp.py report` supersedes them itself "
                                           "(PR.supersede -> _history/) when the next outcome is reported"},
       "premise_check": {"ruling_premise": "the foreign process is gone", "foreign_pid_8628_present": host["foreign_pid_present"],
                         "same_job_running_under_a_new_pid": same_job,
                         "finding": ("pid %d is gone, but the same command line (python -u scripts/m3b_run.py --stage screen --threads 8) is running again as a new "
                                     "instance (instances so far: 8628 started 2026-09-13 20:29 local; 6620 started 2026-09-14 14:54; %s); the host is contended "
                                     "(available %.2f GB, pages_in/s %s, swap %.2f GB) -> the standing hygiene requirement still gates the launch; nothing is launched "
                                     "at re-arm; the foreign process is never signalled" % (
                                         W.FOREIGN_PID, ", ".join("%d started %s" % (p["pid"], p["started"]) for p in same_job) or "none now",
                                         host["host_available_gb"], host["pages_input_per_sec_mean"], host["swap_used_gb"])) if same_job else
                                    ("pid %d is gone and no python process over 1 GB RSS is present; host available %.2f GB, pages_in/s %s, swap %.2f GB; CLEAN = %s" % (
                                        W.FOREIGN_PID, host["host_available_gb"], host["pages_input_per_sec_mean"], host["swap_used_gb"], host["CLEAN"])),
                         "host_at_rearm": host},
       "what_happens_next": {"arm": "phg_webqsp.py arm -> fresh %.0f h wait budget (logs/phg_webqsp_wait_state.json)" % W.PREFLIGHT["wait_budget_hours"],
                             "watcher": "scratchpad/_phg_webqsp_watch.py (background): one `preflight` sample per wait-loop interval; on READY (3 consecutive clean samples) "
                                        "runs inputs -> prereg -> run -> report through the frozen program (each launch behind its own formal preflight; rc 3 = contended -> "
                                        "back to waiting); on an exhausted budget runs not_run -> report; it never signals any process",
                             "unchanged": {"program": pin(W.THIS), "mpi_ranks": W.P.NP, "phg_parameters_in_order": W.P.PARAMS, "preflight_criterion": W.PREFLIGHT,
                                           "no_parameter_change": True, "foreign_process_untouched": True}},
       "preflight_state_kept": {"n_samples_so_far": (old_state or {}).get("n"), "note": "sample numbering continues; the streak resets on the first sample"}}
wj(rec_p, rec)
log("re-arm record -> %s" % os.path.relpath(rec_p, REPO))
st = W.arm()
print(json.dumps({"rearm": pin(rec_p), "wait_state": {k: st[k] for k in ("armed_utc", "deadline_utc")}, "superseded_runs": moved_runs, "same_job": same_job,
                  "host": {k: host[k] for k in ("host_available_gb", "pages_input_per_sec_mean", "swap_used_gb", "CLEAN")}}, indent=1))
