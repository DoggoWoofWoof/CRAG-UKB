import subprocess, os, time
DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
pairs = [('21,22','1'),('31,32','4'),('33,34','6'),('35,36','8'),('48,49','10'),('58,59','1'),('60,61','4'),('70,71','6'),('107,108','8'),('109,110','10'),('114,115','1'),('118,119','4')]
for shards, acct in pairs:
    print(f'LAUNCH 2wiki_universe docs dense {shards} acct{acct} batch16', flush=True)
    subprocess.Popen(['python','experiments.py','run','canonical-encode','--backend','modal','--account',acct,'--','--dataset','2wiki_universe','--kind','docs','--model','dense','--shard-ids',shards,'--batch','16'],
                     cwd='C:/Users/Swastik/Desktop/CRAG', creationflags=DETACHED, close_fds=True)
    time.sleep(4)
print('DISPATCHED 12 jobs', flush=True)
