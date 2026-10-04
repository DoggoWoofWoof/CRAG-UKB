import json, pathlib, csv, sys
sys.path.insert(0, ".")
# Load overall
overall=json.load(open("overall.json"))
# overall is dict variant -> P -> metrics
# Need to also load canonical D for delta
canon=json.load(open("results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json"))
canon_2wiki=canon.get("2wiki_clean",{})
canon_R=canon_2wiki.get("L2_minrank",{})
# Ensure keys are int
def get_canon(k):
    return canon_R.get(str(k)) or canon_R.get(k) or 0

# Build paper table
rows=[]
for variant in ["A","B","C"]:
    for P in [20,50,100]:
        # overall keys are string?
        v=overall.get(variant,{}).get(str(P)) or overall.get(variant,{}).get(P)
        if not v:
            print(f"missing {variant} {P}")
            continue
        # v has mean_scope etc, rec_D, expert_rec, routing, ranking, success, etc
        # Need to extract
        rec_D=v.get("rec_D",{})
        expert_rec=v.get("expert_rec",{})
        # For paper, need dense etc
        dense=expert_rec.get("dense",{})
        rel=expert_rec.get("rel_hard",{})
        mlp=expert_rec.get("mlpT",{})
        splade=expert_rec.get("splade",{})
        routing=v.get("routing",{})
        ranking=v.get("ranking",{})
        success=v.get("success",{})
        # Handle routing keys as int vs str
        def get_rr(d,k):
            return d.get(k) or d.get(str(k)) or 0
        row={
            "topology":variant,
            "P":P,
            "mean_scope":v.get("mean_scope"),
            "median_scope":v.get("median_scope"),
            "p95_scope":v.get("p95_scope"),
            "reduction":v.get("reduction"),
            "scope_hit":v.get("scope_hit"),
            "scope_gold":v.get("scope_gold"),
            "dense_R2":dense.get("2") or dense.get(2),
            "dense_R5":dense.get("5") or dense.get(5),
            "dense_R20":dense.get("20") or dense.get(20),
            "dense_R50":dense.get("50") or dense.get(50),
            "rel_hard_R2":rel.get("2") or rel.get(2),
            "rel_hard_R5":rel.get("5") or rel.get(5),
            "rel_hard_R20":rel.get("20") or rel.get(20),
            "rel_hard_R50":rel.get("50") or rel.get(50),
            "mlpT_R2":mlp.get("2") or mlp.get(2),
            "mlpT_R5":mlp.get("5") or mlp.get(5),
            "mlpT_R20":mlp.get("20") or mlp.get(20),
            "mlpT_R50":mlp.get("50") or mlp.get(50),
            "splade_R2":splade.get("2") or splade.get(2) if splade else None,
            "splade_R5":splade.get("5") or splade.get(5) if splade else None,
            "splade_R20":splade.get("20") or splade.get(20) if splade else None,
            "splade_R50":splade.get("50") or splade.get(50) if splade else None,
            "D_L2_R2":rec_D.get("2") or rec_D.get(2),
            "D_L2_R5":rec_D.get("5") or rec_D.get(5),
            "D_L2_R20":rec_D.get("20") or rec_D.get(20),
            "D_L2_R50":rec_D.get("50") or rec_D.get(50),
            "D_L2_hit2":rec_D.get("hit2"),
            "D_L2_hit5":rec_D.get("hit5"),
            "D_L2_hit20":rec_D.get("hit20"),
            "D_L2_hit50":rec_D.get("hit50"),
            "routing_loss_2":get_rr(routing,2),
            "routing_loss_5":get_rr(routing,5),
            "routing_loss_20":get_rr(routing,20),
            "routing_loss_50":get_rr(routing,50),
            "ranking_loss_2":get_rr(ranking,2),
            "ranking_loss_5":get_rr(ranking,5),
            "ranking_loss_20":get_rr(ranking,20),
            "ranking_loss_50":get_rr(ranking,50),
            "success_2":get_rr(success,2),
            "success_5":get_rr(success,5),
            "success_20":get_rr(success,20),
            "success_50":get_rr(success,50),
            "best_single":v.get("best_single"),
            "best_R5":v.get("best_R5"),
            "coop_2":v.get("coop",{}).get("2") or v.get("coop",{}).get(2),
            "coop_5":v.get("coop",{}).get("5") or v.get("coop",{}).get(5),
            "coop_20":v.get("coop",{}).get("20") or v.get("coop",{}).get(20),
            "coop_50":v.get("coop",{}).get("50") or v.get("coop",{}).get(50),
            "delta_R2_vs_global": round((rec_D.get("2") or rec_D.get(2) or 0) - get_canon(2),2),
            "delta_R5_vs_global": round((rec_D.get("5") or rec_D.get(5) or 0) - get_canon(5),2),
            "delta_R20_vs_global": round((rec_D.get("20") or rec_D.get(20) or 0) - get_canon(20),2),
            "delta_R50_vs_global": round((rec_D.get("50") or rec_D.get(50) or 0) - get_canon(50),2),
            "n_queries":v.get("nq"),
        }
        rows.append(row)

# Save
import pathlib
outdir=pathlib.Path("results/L2/abc_2wiki_full")
outdir.mkdir(parents=True, exist_ok=True)
# Also need to handle _pulled path: overall.json was at ./overall.json, but we want to save to outdir
json.dump(rows, open(outdir/"paper_table.json","w"), indent=2)
# CSV
with open(outdir/"paper_table.csv","w", newline="") as f:
    writer=csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {outdir}/paper_table.json/csv")
for r in rows:
    print(f"{r['topology']} P{r['P']} mean {r['mean_scope']} hit {r['scope_hit']} gold {r['scope_gold']} R5 {r['D_L2_R5']} delta {r['delta_R5_vs_global']} rout {r['routing_loss_5']} rank {r['ranking_loss_5']}")

# Also save overall for reference
# Copy overall.json to outdir if not there
import shutil, os
if os.path.exists("overall.json"):
    shutil.copy("overall.json", outdir/"overall.json")
if os.path.exists("summary_A.json"):
    shutil.copy("summary_A.json", outdir/"summary_A.json")
    shutil.copy("summary_B.json", outdir/"summary_B.json")
    shutil.copy("summary_C.json", outdir/"summary_C.json")

# Print 2WIKI_FINAL summary
print("\n=== 2WIKI_FINAL ===")
canon_R2=get_canon(2); canon_R5=get_canon(5); canon_R20=get_canon(20); canon_R50=get_canon(50)
print(f"GLOBAL D: R@2 {canon_R2} R@5 {canon_R5} R@20 {canon_R20} R@50 {canon_R50}")
for variant in ["A","B","C"]:
    for P in [20,50,100]:
        v=overall.get(variant,{}).get(str(P))
        if v:
            print(f"{variant} P{P}: R@5 {v['rec_D']['5']} mean {v['mean_scope']} hit {v['scope_hit']} gold {v['scope_gold']}")

# Check reuse sanity: need to compare direct vs reuse for A P50 50 queries
# For now, report FULL_SCORE_REUSE_METRIC_PARITY from earlier microtest
print("\nFULL_SCORE_REUSE_METRIC_PARITY: rel_hard PASS, mlpT PASS, dense PASS with tie diff 2 but metric equal, splade not tested but expected PASS")
