import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch
import l2_controller as CT
from _run_controllers import save_perq, summarize, OUT
log = lambda *a: print(*a, flush=True)
va = {ds: CT.load_cache(ds, "val") for ds in CT.DS}
# C3 (5 epochs; collapse is flat by ep1 -> init-inclusive selection unaffected by more epochs)
m3, meta3 = CT.train("C3", epochs=5, lr=3e-4, log=log)
r3 = CT.evaluate(m3, "C3", va, want_gate=True, collect=True); save_perq("C3", r3)
torch.save(m3.state_dict(), f"{OUT}/C3.pt"); summarize("C3", r3, meta3)
# C4 (C3 arch + positive-KL Shapley aux)
m4, meta4 = CT.train("C4", epochs=5, lr=3e-4, lam=0.05, log=log)
r4 = CT.evaluate(m4, "C4", va, want_gate=True, collect=True); save_perq("C4", r4)
torch.save(m4.state_dict(), f"{OUT}/C4.pt"); summarize("C4", r4, meta4)
log("C3C4_DONE")
