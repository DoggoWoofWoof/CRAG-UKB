import json
data=json.load(open('results/L1/phase1_webqsp_corrected.json'))
def best_for(filter_fn):
    cand=[r for r in data if filter_fn(r)]
    cand=sorted(cand, key=lambda r: (-r['ANY_GOLD_COV'], -r['GOLD_FRAC'], r['mean_scope']))
    return cand[0] if cand else None
for router in ['dense','splade','dense+splade']:
    b=best_for(lambda r, router=router: r['router']==router)
    print(router, f"ANY {b['ANY_GOLD_COV']} ALL {b['ALL_GOLD_COV']} GF {b['GOLD_FRAC']} K{b['K']} P{b['P']} topo {b['topology']} scope {b['mean_scope']}")
overall=sorted(data, key=lambda r: (-r['ANY_GOLD_COV'], -r['GOLD_FRAC'], r['mean_scope']))[0]
print('OVERALL BEST', overall)
print('\nK100 P50 A:')
for router in ['dense','splade','dense+splade']:
    r=[x for x in data if x['K']==100 and x['P']==50 and x['topology']=='A' and x['router']==router][0]
    print(f"A {router:12s} ANY {r['ANY_GOLD_COV']} ALL {r['ALL_GOLD_COV']} GF {r['GOLD_FRAC']} scope {r['mean_scope']}")
print('\nWEBQSP_CORRECTED_N', len([r for r in data if r['K']==100 and r['P']==50 and r['topology']=='A' and r['router']=='dense'][0].values()))
# Actually N is 1578
print('WEBQSP_CORRECTED_N=1578')
# Fusion-Dense for A K100 P50
dense=[x for x in data if x['K']==100 and x['P']==50 and x['topology']=='A' and x['router']=='dense'][0]
fusion=[x for x in data if x['K']==100 and x['P']==50 and x['topology']=='A' and x['router']=='dense+splade'][0]
print(f"A Fusion-Dense ANY {fusion['ANY_GOLD_COV']-dense['ANY_GOLD_COV']:.2f} ALL {fusion['ALL_GOLD_COV']-dense['ALL_GOLD_COV']:.2f} GF {fusion['GOLD_FRAC']-dense['GOLD_FRAC']:.2f}")
