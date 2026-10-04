import json
data=json.load(open('results/L1/phase1_hotpot.json'))
# hotpot dense only, 36 configs
def best_for(filter_fn):
    cand=[r for r in data if filter_fn(r)]
    cand=sorted(cand, key=lambda r: (-r['ALL_GOLD_COV'], -r['GOLD_FRAC'], r['mean_scope']))
    return cand[0] if cand else None
# Since only dense, overall best is dense
overall=sorted(data, key=lambda r: (-r['ALL_GOLD_COV'], -r['GOLD_FRAC'], r['mean_scope']))[0]
print('HOTPOT DENSE ONLY 36 cells')
print('OVERALL BEST DENSE', overall)
print('\nK100 P50 table dense:')
for topo in ['A','B','C']:
    r=[x for x in data if x['K']==100 and x['P']==50 and x['topology']==topo and x['router']=='dense'][0]
    print(f"{topo} dense ALL {r['ALL_GOLD_COV']} ANY {r['ANY_GOLD_COV']} GF {r['GOLD_FRAC']} scope {r['mean_scope']}")
# Also show P saturation for best dense
best=overall
print('\nBest dense config', best)
for K in [25,50,100,200]:
    r=[x for x in data if x['K']==K and x['P']==best['P'] and x['router']=='dense' and x['topology']==best['topology']][0]
    print(f"K{K} ALL {r['ALL_GOLD_COV']} ANY {r['ANY_GOLD_COV']} scope {r['mean_scope']}")
for P in [20,50,100]:
    r=[x for x in data if x['K']==best['K'] and x['P']==P and x['router']=='dense' and x['topology']==best['topology']][0]
    print(f"P{P} ALL {r['ALL_GOLD_COV']} scope {r['mean_scope']}")
# Topo delta at K100 P50 dense
a=[x for x in data if x['K']==100 and x['P']==50 and x['topology']=='A' and x['router']=='dense'][0]
b=[x for x in data if x['K']==100 and x['P']==50 and x['topology']=='B' and x['router']=='dense'][0]
c=[x for x in data if x['K']==100 and x['P']==50 and x['topology']=='C' and x['router']=='dense'][0]
print(f"\nB-A {b['ALL_GOLD_COV']-a['ALL_GOLD_COV']:.2f} C-A {c['ALL_GOLD_COV']-a['ALL_GOLD_COV']:.2f} C-B {c['ALL_GOLD_COV']-b['ALL_GOLD_COV']:.2f}")
