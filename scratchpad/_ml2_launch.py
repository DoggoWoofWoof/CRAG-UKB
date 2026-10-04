"""FBX_SCALE stage 4C -- launch one host CELL job per candidate of addendum 5's grid (each job builds the candidate's attempted K cells under the host-yield wrapper; a cell that already has a
record is skipped, so a job killed by a higher-priority job is simply relaunched).

  python scratchpad/_ml2_launch.py            # print the commands
  python scratchpad/_ml2_launch.py --go       # launch the ones whose cells are not all recorded yet (records are checked in the LOCAL results/FREEBASE_SCALE/ml2 -- fetch first)
"""
import io
import json
import os
import subprocess
import sys

ROOT = "C:/Users/Swastik/Desktop/CRAG"
RX = "C:/Users/Swastik/Desktop/message-passing-retrieval/tools/rx/rx.py"
ADD = ROOT + "/results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_5.json"
ML2 = ROOT + "/results/FREEBASE_SCALE/ml2"


def main(go, mem="24", skip=()):
    g = json.load(io.open(ADD, encoding="utf-8"))["candidates"]["grid"]
    n = 0
    for c in g:
        if c["candidate"] in skip:
            continue
        ks =[K for K in c["attempted_at_K"] if not (os.path.exists("%s/webqsp__%s_k%d.RUN.json" % (ML2, c["candidate"], K)) or os.path.exists("%s/webqsp__%s_k%d.FAILED.json" % (ML2, c["candidate"], K)))]
        if not ks:
            continue
        cmd = [sys.executable, RX, "run", "-n", "fbx-ml2-cell-%s" % c["candidate"].replace("_", "-"), "--cpus", "4", "--mem", mem, "--","python", "-u", "scratchpad/_host_yield.py", "run", "--",
               "python", "-u", "scratchpad/_ml2_cell.py", "CELL", c["method"], str(c["Wmax"]), str(c["level"]), ",".join(str(k) for k in ks)]
        n += 1
        print(" ".join(cmd[2:]))
        if go:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            print("  ->", (r.stdout.strip().splitlines() or ["(no output)"])[-1], "" if r.returncode == 0 else "RC %d %s" % (r.returncode, r.stderr[-300:]))
    print("%d job(s)" % n)


if __name__ == "__main__":
    a = sys.argv[1:]
    main("--go" in a, mem=next((x.split("=", 1)[1] for x in a if x.startswith("--mem=")), "24"),
         skip=tuple(next((x.split("=", 1)[1].split(",") for x in a if x.startswith("--skip=")), [])))
