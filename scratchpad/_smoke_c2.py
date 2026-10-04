import sys; sys.path.insert(0,'scratchpad')
import l2_controller as CT
m,meta=CT.train('C2',epochs=8,lr=3e-4,log=print)
print('C2 BEST_EPOCH',meta['best_epoch'],'BEST_SEL',meta['best_sel'])
