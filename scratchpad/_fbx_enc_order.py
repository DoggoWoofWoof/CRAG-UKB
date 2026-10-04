"""Encode ORDER only (addendum 13): encode a 32-chunk stratified sample of the Freebase PQ64 chunks first, then everything else in the usual order.

Nothing about a chunk changes: every chunk is produced by _fbx_encode.cmd_chunks itself (same model, contract, codebook, batch composition, quantiser, record and checksum), and a chunk whose record + checksum verify is skipped, so
the final set of 1,872 chunk files is the one `CHUNKS 0 1872` would have produced.  The only thing this wrapper changes is which chunks are still missing when cmd_chunks runs: phase 1 hides every chunk outside the sample from
cmd_chunks (chunk_ok answers True for it), phase 2 restores chunk_ok and encodes whatever is left, ascending.  It is resumable at any point (a killed job relaunched with the same command continues).

The sample: 32 strata of equal width over the 1,872 chunks (58.5 chunks each), the chunk at the midpoint of each stratum.  Chunk c holds rows [c * 131072, (c + 1) * 131072) in node first-occurrence order, so the strata spread the
sample over the whole name space, which a prefix of chunks does not.  The nested sizes T2 will use (1, 2, 4, 8, 16, 32) are taken from this list in the bit-reversed stratum order SAMPLE_ORDER below (every prefix of 2^j is an even spread).

  python -u scratchpad/_fbx_enc_order.py [LIST]          # LIST: print the sample and exit
  python -u scratchpad/_fbx_enc_order.py RETRY <tries>   # the job form: the ordering above, retried like _fbx_enc_run.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fbx_encode as E  # noqa: E402

NCH = 1872
NS = 32
SAMPLE_BY_STRATUM = [int((k + 0.5) * NCH / NS) for k in range(NS)]


def bitrev(k, bits=5):
    return int(format(k, "0%db" % bits)[::-1], 2)


SAMPLE_ORDER = [SAMPLE_BY_STRATUM[bitrev(k)] for k in range(NS)]
assert sorted(SAMPLE_ORDER) == SAMPLE_BY_STRATUM and len(set(SAMPLE_ORDER)) == NS and max(SAMPLE_ORDER) < NCH - 1


def main():
    c, _ = E.load_contract()
    nch = int(c["ordering"]["n_chunks"])
    assert nch == NCH, nch
    S = set(SAMPLE_BY_STRATUM)
    orig = E.chunk_ok

    def only_sample(i, csha, cbsha):
        return orig(i, csha, cbsha) if i in S else True

    E.log("phase 1: stratified sample, %d chunks: %s" % (NS, SAMPLE_BY_STRATUM))
    E.chunk_ok = only_sample
    try:
        E.cmd_chunks(0, nch)
    finally:
        E.chunk_ok = orig
    E.log("phase 2: every remaining chunk, ascending")
    E.cmd_chunks(0, nch)


def retry(tries):
    """Same orchestration as _fbx_enc_run.py (a device-wide CUDA fault kills the encode; the run is resumable, a retry costs at most the chunk in flight), around THIS ordering."""
    import subprocess
    import time
    rc = 1
    for i in range(1, tries + 1):
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__)])
        print("[enc_order] try %d/%d rc %d after %.0f s" % (i, tries, rc, time.time() - t0), flush=True)
        if rc == 0:
            return 0
        time.sleep(60)
    return rc


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["LIST"]:
        print("by stratum:", SAMPLE_BY_STRATUM)
        print("nested order:", SAMPLE_ORDER)
    elif len(a) == 2 and a[0] == "RETRY":
        sys.exit(retry(int(a[1])))
    else:
        main()
