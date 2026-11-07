"""Print a structural summary of a JSON file (keys, types, sizes, short samples).

Usage: python3 -I peek.py <json path> [max_depth] [max_list_items]
"""
import json
import sys


def short(v, n=120):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "..."


def walk(obj, depth, maxd, maxl, indent=""):
    if isinstance(obj, dict):
        print(f"{indent}dict[{len(obj)}]")
        if depth >= maxd:
            print(f"{indent}  keys: {list(obj.keys())[:40]}")
            return
        for i, (k, v) in enumerate(obj.items()):
            if i >= 60:
                print(f"{indent}  ... ({len(obj) - 60} more keys)")
                break
            if isinstance(v, (dict, list)):
                print(f"{indent}  {k!r}:")
                walk(v, depth + 1, maxd, maxl, indent + "    ")
            else:
                print(f"{indent}  {k!r}: {short(v)}")
    elif isinstance(obj, list):
        print(f"{indent}list[{len(obj)}]")
        if depth >= maxd:
            if obj:
                print(f"{indent}  first: {short(obj[0])}")
            return
        for i, v in enumerate(obj[:maxl]):
            print(f"{indent}  [{i}]:")
            walk(v, depth + 1, maxd, maxl, indent + "    ")
    else:
        print(f"{indent}{short(obj)}")


def main():
    path = sys.argv[1]
    maxd = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    maxl = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    walk(obj, 0, maxd, maxl)


if __name__ == "__main__":
    main()
