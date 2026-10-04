import json
data=json.load(open('results/L1/phase1_musique.json'))
for P in [20,50,100]:
    r=[x for x in data if x['K']==100 and x['P']==P and x['topology']=='C' and x['router']=='dense+splade'][0]
    print(f"C dense+splade K100 P{P} ALL {r['ALL_GOLD_COV']} ANY {r['ANY_GOLD_COV']} GF {r['GOLD_FRAC']} scope {r['mean_scope']} frac {r['scope_frac']} red {r['reduction']}")
