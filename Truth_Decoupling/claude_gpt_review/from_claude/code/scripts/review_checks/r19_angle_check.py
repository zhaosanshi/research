#!/usr/bin/env python3
"""Read-only unsigned initial/final axis angles; malformed vectors fail loudly."""
import argparse,math,sys
from pathlib import Path
from r19_validation import checkpoint,layer_vectors,cosine


def check(root):
    for model in ('llama3.1-8b','qwen3-8b'):
        for init in ('das','random','md'):
            path=root/model/'ce_alpha3_ep3'/init/'adapter.pt';a=checkpoint(path)
            before=layer_vectors(a,'initial_W',path);after=layer_vectors(a,'W',path)
            if set(before)!=set(after):raise ValueError('initial/final layer sets differ: '+str(path))
            for layer in sorted(after):
                c=abs(cosine(before[layer],after[layer]));angle=math.degrees(math.acos(c))
                print('%-12s %-7s L%s unsigned axis angle %.2f deg (|cos| %.4f)'%(model,init,layer,angle,c))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    try:return check(a.root)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print('INVALID INPUT: '+str(exc),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
