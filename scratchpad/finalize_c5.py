from src.experiments.canonical_encode import finalize
import time, json
for ds,kind,model in [('2wiki_universe','docs','dense'),('hotpotqa','docs','dense')]:
    s=time.time()
    c,p,ns=finalize(ds,kind,model)
    print(f'{ds} {kind} {model}: complete={c} {len(p)}/{ns} secs {time.time()-s:.1f}')
    m=json.load(open(f'data/canonical/{ds}/encodings/{model}/{kind}/manifest.json'))
    print(f'  manifest complete {m["complete"]} rows {m["rows_covered"]}/{m["n_items"]} sha {m["source_sha256"][:8]}')
    # deep validate
    import os, numpy as np
    od=f'data/canonical/{ds}/encodings/{model}/{kind}'
    total=0
    for idx in m['shards_present']:
        ids=json.load(open(f'{od}/ids_{idx:05d}.json'))
        arr=np.load(f'{od}/shard_{idx:05d}.npy', mmap_mode='r')
        assert arr.shape[0]==len(ids) and arr.shape[1]==1536 and arr.dtype==np.float16
        total+=len(ids)
    print(f'  deep validated {len(m["shards_present"])} shards {total} rows dim 1536 fp16')
