import os
R = "C:/Users/Swastik/Desktop/CRAG/data/final_canonical/webqsp"
D = "C:/Users/Swastik/Desktop/CRAG/data/_cache/fbx_stage/webqsp"
assert not os.path.isdir("C:/Users/Swastik/Desktop/CRAG/data/_cache/fbx_stage") or not os.listdir("C:/Users/Swastik/Desktop/CRAG/data/_cache/fbx_stage")
n = b = 0
for ln in open("C:/Users/Swastik/Desktop/CRAG/data/_cache/webqsp_missing.txt", encoding="utf-8"):
    sz, p = ln.rstrip("\n").split("\t", 1)
    d = os.path.join(D, os.path.dirname(p))
    os.makedirs(d, exist_ok=True)
    os.link(os.path.join(R, p), os.path.join(D, p))
    n += 1; b += int(sz)
print("staged", n, "files", b, "bytes")
