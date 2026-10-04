import torch, hashlib, os
for fname in ['head_1133e70759e71589.pt','head_0e7c0fe495f65776.pt','head_06a9fd3a3e39b3d0.pt','head_32404bf9b65a2d95.pt']:
    p=f'data/ukb_storage/_head_cache/{fname}'
    if not os.path.exists(p):
        print(fname, 'missing')
        continue
    sd=torch.load(p, map_location='cpu')
    keys=list(sd.keys())
    dim=None
    K=None
    if 'net.0.weight' in sd:
        dim=sd['net.0.weight'].shape[1]
    if 'net.2.weight' in sd:
        K=sd['net.2.weight'].shape[0]//1536 if sd['net.2.weight'].shape[0] % 1536 ==0 else sd['net.2.weight'].shape[0]
        # Actually for Mix, shape is [12288,512] where 12288 = 8*1536
        try:
            K = sd['net.2.weight'].shape[0] // 1536
        except:
            K = sd['net.2.weight'].shape[0]
    print(f'{fname} size {os.path.getsize(p)} SHA {hashlib.sha256(open(p,"rb").read()).hexdigest()[:16]} dim {dim} K {K} keys {keys[:3]}')
    for k in ['net.0.weight','net.2.weight','net.0.bias']:
        if k in sd:
            print(f'  {k} {tuple(sd[k].shape)}')
