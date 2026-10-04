"""Generic summary printer for audit JSON files -- strips per-query indicator arrays
(any key starting with "_ind", or any list longer than 20 items) so these can be
inspected without blowing up context with 2000-element boolean arrays.

  python scratchpad/_l1au_view.py <path1> [path2 ...]
  python scratchpad/_l1au_view.py glob "<pattern>"
"""
import sys, json, glob as globmod


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_ind")}
    if isinstance(o, list):
        if len(o) > 20:
            return "<list len=%d, elided>" % len(o)
        return [strip(v) for v in o]
    return o


def show(fp):
    print("\n===== %s =====" % fp)
    d = json.load(open(fp))
    print(json.dumps(strip(d), indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "glob":
        for fp in sorted(globmod.glob(a[1])):
            show(fp)
    else:
        for fp in a:
            show(fp)
