import time, json
out = {}
t=time.time()
from sentence_transformers import SentenceTransformer
out['import_s']=round(time.time()-t,1)
t=time.time()
m=SentenceTransformer('Alibaba-NLP/gte-Qwen2-1.5B-instruct', trust_remote_code=True)
out['load_s']=round(time.time()-t,1); out['device']=str(m.device)
json.dump(out, open('scratchpad/_enc_probe.json','w'))
for bs in (8,32,64):
    sents=['The film was directed by Stuart Rosenberg and released in 1970.']*bs
    t=time.time(); e=m.encode(sents, normalize_embeddings=True, show_progress_bar=False, batch_size=bs)
    out[f'encode_{bs}_s']=round(time.time()-t,2); out[f'ms_per_sent_bs{bs}']=round((time.time()-t)/bs*1000,1)
    out['dim']=int(e.shape[1])
    json.dump(out, open('scratchpad/_enc_probe.json','w'))
out['DONE']=True
json.dump(out, open('scratchpad/_enc_probe.json','w'))
print(json.dumps(out))
