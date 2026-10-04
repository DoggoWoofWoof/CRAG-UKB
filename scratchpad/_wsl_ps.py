"""List Zoltan/mpirun processes in the host's WSL distro (read only):  python scratchpad/_wsl_ps.py"""
import subprocess

out = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--", "ps", "-eo", "pid,etimes,args"], capture_output=True, text=True).stdout
rows = [ln[:180] for ln in out.splitlines() if ("phg_driver" in ln or "mpirun" in ln) and "ps -eo" not in ln]
print("\n".join(rows) if rows else "no phg_driver / mpirun processes")
