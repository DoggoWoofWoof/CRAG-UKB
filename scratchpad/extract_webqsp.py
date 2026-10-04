import json
data=json.load(open('results/L1/phase1_webqsp.json'))
def best_for(filter_fn):
    cand=[r for r in data if filter_fn(r)]
    cand=sorted(cand, key=lambda r: (-r['ANY_GOLD_COV'], -r['GOLD_FRAC'], r['mean_scope']))
    return cand[0] if cand else None
for router in ['dense','splade','dense+splade']:
    b=best_for(lambda r, router=router: r['router']==router)
    print(router, b)
overall=sorted(data, key=lambda r: (-r['ANY_GOLD_COV'], -r['GOLD_FRAC'], r['mean_scope']))[0]
print('OVERALL BEST', overall)
print('\nK100 P50 table:')
for topo in ['A','B','C']:
    for router in ['dense','splade','dense+splade']:
        r=[x for x in data if x['K']==100 and x['P']==50 and x['topology']==topo and x['router']==router][0]
        print(f"{topo} {router:12s} ANY {r['ANY_GOLD_COV']} ALL {r['ALL_GOLD_COV']} GF {r['GOLD_FRAC']} scope {r['mean_scope']}")
best=overall
print('\nBest config', best)
for K in [25,50,100,200]:
    r=[x for x in data if x['K']==K and x['P']==best['P'] and x['router']==best['router'] and x['topology']==best['topology']][0]
    print(f"K{K} ANY {r['ANY_GOLD_COV']} scope {r['mean_scope']}")
print('P saturation for best K')
for P in [20,50,100]:
    r=[x for x in data if x['K']==best['K'] and x['P']==P and x['router']==best['router'] and x['topology']==best['topology']][0]
    print(f"P{P} ANY {r['ANY_GOLD_COV']} scope {r['mean_scope']}")
print('\nDeltas at K100 P50:')
for topo in ['A','B','C']:
    dense=[x for x in data if x['K']==100 and x['P']==50 and x['topology']==topo and x['router']=='dense'][0]
    fusion=[x for x in data if x['K']==100 and x['P']==50 and x['topology']==topo and x['router']=='dense+splade'][0]
    print(f"{topo} Fusion-Dense ANY {fusion['ANY_GOLD_COV']-dense['ANY_GOLD_COV']:.2f} GF {fusion['GOLD_FRAC']-dense['GOLD_FRAC']:.2f}")
for router in ['dense','splade','dense+splade']:
    a=[x for x in data if x['K']==100 and x['P']==50 and x['topology']=='A' and x['router']==router][0]
    b=[x for x in data if x['K']==100 and x['P']==50 and x['topology']=='B' and x['router']==router][0]
    c=[x for x in data if x['K']==100 and x['P']==50 and x['topology']=='C' and x['router']==router][0]
    print(f"{router} B-A {b['ANY_GOLD_COV']-a['ANY_GOLD_COV']:.2f} C-A {c['ANY_GOLD_COV']-a['ANY_GOLD_COV']:.2f} C-B {c['ANY_GOLD_COV']-b['ANY_GOLD_COV']:.2f}")
