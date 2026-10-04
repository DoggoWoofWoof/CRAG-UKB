import subprocess, time, os
shard_pairs = [('21,22','1'),('31,32','4'),('33,34','6'),('35,36','8'),('48,49','10'),('58,59','1'),('60,61','4'),('70,71','6'),('107,108','8'),('109,110','10'),('114,115','1'),('118,119','4')]
import subprocess
DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
for shards, acct in shard_pairs:
    print(f'Launching 2wiki_universe docs dense shards {shards} on account {acct} batch 16', flush=True)
    cmd = ['python','experiments.py','run','canonical-encode','--backend','modal','--account',acct,'--','--dataset','2wiki_universe','--kind','docs','--model','dense','--shard-ids',shards,'--batch','16']
    subprocess.Popen(cmd, cwd='C:/Users/Swastik/Desktop/CRAG', creationflags=DETACHED, close_fds=True)
    time.sleep(5)
print('All 12 OOM recovery jobs dispatched at batch 16', flush=True)
# also try to pull ids for 98,111,132 in parallel (these are invalid-present shards)
for sid in [98,111,132]:
    for ext in [f'ids_{sid:05d}.json', f'shard_{sid:05d}.npy']:
        cmd = ['python','-m','modal','volume','get','crag-data-volume',f'data/canonical/2wiki_universe/encodings/dense/docs/{ext}', 'data/canonical/2wiki_universe/encodings/dense/docs','--force']
        print(f'Pulling {ext}', flush=True)
        subprocess.Popen(cmd, cwd='C:/Users/Swastik/Desktop/CRAG', creationflags=DETACHED, close_fds=True)
        time.sleep(2)
print('Pulls for 98,111,132 dispatched', flush=True)
