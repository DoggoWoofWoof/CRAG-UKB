"""Orchestration only: stream NEW log lines of ONE rx job that match a regex, then its terminal state (each line printed once).

  python -u scratchpad/_watch_job_lines.py <job-id-prefix> <regex> <max-minutes> [poll-seconds]
"""
import re
import subprocess
import sys
import time

RX = ["python", "C:/Users/Swastik/Desktop/message-passing-retrieval/tools/rx/rx.py"]


def run(args):
    return subprocess.run(RX + args, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def main():
    job, pat, mins = sys.argv[1], re.compile(sys.argv[2]), float(sys.argv[3])
    poll = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0
    seen = set()
    t_end = time.time() + 60 * mins
    while time.time() < t_end:
        for line in run(["logs", job]).splitlines():
            if pat.search(line) and line not in seen:
                seen.add(line)
                print(line[:260], flush=True)
        rows = run(["status", job]).splitlines()
        state = rows[1].split()[1] if len(rows) > 1 and len(rows[1].split()) > 1 else "?"
        if state not in ("running", "queued"):
            print("job %s state: %s" % (job, state), flush=True)
            return
        time.sleep(poll)


if __name__ == "__main__":
    main()
