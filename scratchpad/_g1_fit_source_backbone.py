"""Fit the FROZEN source backbone LOCALLY (dev substrate is local) and dump a tiny artifact bundle so the
target-side G1 eval can run wherever the target substrate lives (e.g. Modal) WITHOUT uploading the dev substrate.

Outputs -> results/GENERALIZATION/_g1_backbone/:
  base_full.joblib        C7b soft-archetype head (clf, mu, sd, meta)   [tiny]
  c8c.joblib              C8c XGBRanker reranker                        [~2 MB]
  C11_models.joblib       source C11a (+ emean/estd)  [copied]         [~12 MB]
  source_val_stats.json   source VAL reference norms for feature_compat [<1 KB]
This is the EXACT same fit path as _g1_eval_target.main() fit-mode; only difference is we persist it.
"""
import sys, os, json, shutil
sys.path.insert(0, "scratchpad")
import joblib
import l2_c8 as C8, l2_c9 as C9
from _run_c9 import combined, fit_c8c
import _g1_eval_target as EV

SRC = list(C8.DS); OUT = C8.OUT
BDIR = "results/GENERALIZATION/_g1_backbone"; os.makedirs(BDIR, exist_ok=True)
log = lambda *a: print("[fit]", *a, flush=True)

full = {ds: C8.precompute_arch(ds, "train")["qi"] for ds in SRC}
devinner = {ds: C8.load_split(ds)[1] for ds in SRC}
base_full = C8.fit_soft_head("ndcg50", full)
B_tr = combined("train", full, base_full); B_di = combined("train", devinner, base_full)
c8c = fit_c8c(B_tr["X28"], B_tr["y"], B_tr["groups"], B_di["X28"], B_di["y"], B_di["groups"])
log(f"backbone fit: C7b full + C8c best_iter={c8c.best_iteration}")

# source VAL reference (SRC[0]) -> tiny norm stats
src_ds = SRC[0]
svqi = C8.precompute_arch(src_ds, "val")["qi"]; Bsv = C9.build_bundle(src_ds, "val", svqi, base_full)
ssv = c8c.predict(Bsv["X28"]).astype("float64"); sp_src = EV.build_sp(src_ds, "val", Bsv, ssv)
src_ref = EV.src_ref_from_bundle(sp_src)

joblib.dump(base_full, f"{BDIR}/base_full.joblib")
joblib.dump(c8c, f"{BDIR}/c8c.joblib")
shutil.copy2(f"{OUT}/C11_models.joblib", f"{BDIR}/C11_models.joblib")
json.dump(src_ref, open(f"{BDIR}/source_val_stats.json", "w"), indent=1)
sz = {f: round(os.path.getsize(f"{BDIR}/{f}") / 1e6, 3) for f in os.listdir(BDIR)}
log(f"WROTE {BDIR} sizes_MB={sz}")
log(f"source_val_ref={src_ref}")
print("G1_BACKBONE_FIT_RC=0", flush=True)
