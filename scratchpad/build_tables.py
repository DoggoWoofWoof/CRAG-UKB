import json
hist=json.load(open('results/L2/e2e_full6_universal_gte_qwen.json'))
new=json.load(open('results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json'))
repro=json.load(open('results/L2/e2e_pipeline_gte_qwen_metaqa_2000_repro.json'))
datasets=['musique_clean','2wiki_clean','squad_clean','metaqa','hotpotqa_clean','webqsp']
print('TABLE A D_L2')
for ds in datasets:
    hr=hist[ds]['L2_minrank']
    nr=new[ds]['L2_minrank']
    print(ds)
    for k in [2,5,20,50]:
        print(f" R@{k}: hist {hr[str(k)]:5.1f} new {nr[str(k)]:5.1f} delta {nr[str(k)]-hr[str(k)]:+5.1f}")
print('TABLE B D_E2E')
for ds in datasets:
    hr=hist[ds]['L2_plus_nerL3']
    nr=new[ds]['L2_plus_nerL3']
    print(ds)
    for k in [2,5,20,50]:
        print(f" R@{k}: hist {hr[str(k)]:5.1f} new {nr[str(k)]:5.1f} delta {nr[str(k)]-hr[str(k)]:+5.1f}")
print('METAQA 2000 repro vs hist')
print(f"hist D_L2 50.98 vs repro {repro['metaqa']['L2_minrank']['5']} delta {repro['metaqa']['L2_minrank']['5']-50.98:+.2f}")
print(f"hist D_E2E 68.75 vs repro {repro['metaqa']['L2_plus_nerL3']['5']} delta {repro['metaqa']['L2_plus_nerL3']['5']-68.75:+.2f}")
for k in [2,5,20,50]:
    print(f"k={k} hist L2 {hist['metaqa']['L2_minrank'][str(k)]} repro {repro['metaqa']['L2_minrank'][str(k)]} E2E hist {hist['metaqa']['L2_plus_nerL3'][str(k)]} repro {repro['metaqa']['L2_plus_nerL3'][str(k)]}")
