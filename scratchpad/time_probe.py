import time, torch
t0=time.time()
from src.experiments.kg_hybrid import _load, _pooled_musd
from src.experiments import crag_fusion as CF
from src.experiments import crag_gates as G
DEV=torch.device("cpu")
data=_load(["webqsp","metaqa","squad_clean"],DEV)
mu,sd=_pooled_musd(data,["webqsp","metaqa"],DEV,"train")
splits=G.make_splits(data,["webqsp","metaqa"],0.15,1234)
TR=G._prep_from_idx(data,["webqsp","metaqa"],splits,"opt",mu,sd)
TEv=G._prep_from_idx(data,["webqsp","metaqa"],splits,"val",mu,sd)
refs=G._val_refs(TEv,["webqsp","metaqa"])
print("load", round(time.time()-t0,1),"s")
# time 500 training steps for arm I
t1=time.time()
ck=[]
G._run_confirm_arm(0,"I",TR,TEv,TEv,["webqsp","metaqa"],refs,500,500,0.2,ckpt_sink=ck)
dt=time.time()-t1
print(f"500 steps + 1 val: {dt:.1f}s -> 16000 steps ~ {dt*32/60:.1f} min/arm (train only, excl squad scoring)")
# time scoring one ckpt on squad test
sq=[CF.prepbatch(r,mu,sd,DEV,"squad_clean") for r in data["squad_clean"]["test"]]
t2=time.time()
m=G.GateModel(G.ARMS["I"]["use_int"]); m.load_state_dict(ck[0]["state"]["model"])
G.eval_gate(m,sq,["squad_clean"],"I")
print(f"score 1 ckpt on squad_test(1982): {time.time()-t2:.1f}s -> x32 ckpts x3 arms ~ {(time.time()-t2)*32*3/60:.1f} min")
