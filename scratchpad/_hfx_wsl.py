"""Host housekeeping -- run scratchpad/_hfx.py (PLAN / UPLOAD / FETCH, stdlib-only on the host side) INSIDE the host's WSL distro, where the big Freebase files live on ext4 (/home/student2/crag_ooc: fb CSR, fb_l1, fb_work),
because a Windows rx job cannot see that tree.  Plans / states / url files stay in the rx workspace (reached as /mnt/c/...), the scratch shard goes to the ext4 (not the 9p mount).
   python -u scratchpad/_hfx_wsl.py PLAN <tag> wsl_fb_l1 wsl_fb_work ...       (groups with root "wsl" -- see _hfx.GROUPS)
   python -u scratchpad/_hfx_wsl.py UPLOAD <tag> [threads]
Any environment variable named HFX_* set on the Windows side is passed through, HFX_WS/HFX_STATE/HFX_PACK/HFX_ROOT are set here."""
import os
import subprocess
import sys

DISTRO = os.environ.get("FBX_WSL_DISTRO", "Ubuntu-24.04")
WSL_ROOT = "/home/student2/crag_ooc"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def wp(p):
    p = os.path.abspath(p).replace("\\", "/")
    return "/mnt/" + p[0].lower() + p[2:] if len(p) > 1 and p[1] == ":" else p


env = {k: v for k, v in os.environ.items() if k.startswith("HFX_")}
env.update({"HFX_WS": WSL_ROOT, "HFX_STATE": wp(REPO), "HFX_PACK": WSL_ROOT + "/_pack", "HFX_ROOT": "wsl"})
argv = list(sys.argv[1:])
if "--known" in argv:                                                    # a workspace path -> the same file as /mnt/c/... inside WSL
    i = argv.index("--known")
    argv[i + 1] = wp(argv[i + 1])
cmd = ["wsl.exe", "-d", DISTRO, "--exec", "env"] + ["%s=%s" % kv for kv in env.items()] + ["python3", "-u", wp(os.path.join(HERE, "_hfx.py"))] + argv
sys.exit(subprocess.call(cmd))
