"""Peek helper (read-only): first line prefixes of NSM dev/train files (never test) + line counts via binary read."""
import os, sys, json
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSM = os.path.join(ROOT, "data", "final_canonical", "webqsp", "_acquisition", "nsm", "extracted")
for name, sub in (("webqsp", "webqsp/webqsp"), ("cwq", "CWQ/CWQ")):
    for split in ("dev_simple.json", "train_simple.json"):
        p = os.path.join(NSM, sub, split)
        with open(p, "rb") as f:
            head = f.read(1500).decode("utf-8", "replace")
        print(name, split, os.path.getsize(p))
        print(head[:1200].replace("\n", "\\n"))
        print("---")
