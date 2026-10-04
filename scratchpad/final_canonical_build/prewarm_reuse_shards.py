"""Pre-warm ONLY step 1 of reuse_map_kb.py: the per-shard token-ID digest cache.

    python scratchpad/final_canonical_build/prewarm_reuse_shards.py [2wiki|hotpotqa ...] --workers N

Runs exactly reuse_map_kb.work_shard over exactly the same job list and writes exactly the same
{WORKD}/{ds}/{FAMILY}_{shard}.npz files, which the real run then skips (work_shard is cached on the
output path).  Nothing else is computed and nothing outside data/final_canonical/_work/ is touched.
This exists so the dominant tokenization cost can overlap with the query-independence rebuilds
without also holding the full run's ~2 GB of parent-side arrays in RAM at the same time.
"""
import os, sys, glob, time, argparse, multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reuse_map_kb as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", nargs="*", default=["2wiki", "hotpotqa"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    pool = mp.Pool(a.workers, initializer=R._init)
    try:
        for ds in a.ds:
            W = f"{R.WORKD}/{ds}"
            os.makedirs(W, exist_ok=True)
            for name, pattern, _encdir, _parse in R.FAMILIES[ds]:
                shards = sorted(glob.glob(pattern))
                jobs = [(p, f"{W}/{name}_{os.path.basename(p)}.npz") for p in shards]
                todo = [j for j in jobs if not os.path.exists(j[1])]
                print(f"[{ds}] {name}: {len(shards)} shards, {len(todo)} to tokenize", flush=True)
                done = rows = 0
                for _op, n in pool.imap_unordered(R.work_shard, todo):
                    done += 1; rows += n
                    if done % 10 == 0 or done == len(todo):
                        print(f"[{ds}] {name}: {done}/{len(todo)} shards, {rows} rows, "
                              f"{time.time()-t0:.0f}s", flush=True)
    finally:
        pool.close(); pool.join()
    print(f"prewarm done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
