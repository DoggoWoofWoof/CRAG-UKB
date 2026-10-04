"""C5 = C4 arch (query+candidate gate) + Shapley aux + EXPERT DROPOUT.
Justified: C2/C3 collapse (trained gate drifts to 0.7815 full-scope NDCG, MRR up but deep-recall
specialists suppressed) => 'specialists ignored'. Expert dropout forces the gate to not rely on any
single expert. VAL only. Saves C5.json/pt/_perq."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch
import l2_controller as CT
from _run_controllers import OUT, save_perq, summarize, log

def main():
    va = {ds: CT.load_cache(ds, "val") for ds in CT.DS}
    T0 = time.time()
    m5, meta5 = CT.train("C4", epochs=5, lr=3e-4, lam=0.05, expdrop=0.3, log=log)
    r5 = CT.evaluate(m5, "C4", va, want_gate=True, collect=True); save_perq("C5", r5)
    torch.save(m5.state_dict(), f"{OUT}/C5.pt"); summarize("C5", r5, meta5)
    log(f"C5_DONE total={time.time()-T0:.0f}s")

if __name__ == "__main__":
    main()
