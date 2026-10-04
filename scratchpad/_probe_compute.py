"""Cheapest possible test of whether a workspace can actually RUN something.

Volume writes and `volume list` keep working on a workspace that has exhausted its compute
credit -- extra_wNzonK and crm both accepted a 4.72 GB store upload and then refused to run a
function.  So liveness has to be probed with compute, not with storage, and probing it with the
real app would stage gigabytes first.  This is one trivial CPU function and nothing else.

  MODAL_PROFILE=<p> modal run scratchpad/_probe_compute.py
"""
import modal

app = modal.App("crag-probe-compute")


@app.function(cpu=1.0, memory=512, timeout=120)
def ping():
    import platform
    return "alive %s" % platform.machine()


@app.local_entrypoint()
def main():
    print("PROBE_RESULT: %s" % ping.remote())
