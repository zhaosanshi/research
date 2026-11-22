"""Read-only baseline and positive/negative orientation tests for the proposed Llama keep loss.
Run on CPU in the scoring container from Truth_Decoupling. This does not train the new direction.
"""
import json
from pathlib import Path
import re
import sys
import unicodedata
import torch

sys.path.insert(0,str(Path("code_e45_20261112").resolve()))
from core.qa_factorial_das import ShiftHook

ROOT = Path("Iteration_01")
FACTS = ("triviaqa","popqa")
SOURCE = Path("results_e44_factorial_das/llama3.1-8b_side_E2")


def norm(text):
    return re.sub(r"[^\w]+"," ",unicodedata.normalize("NFKC",text).casefold()).strip()


def check_questions():
    banks={}
    for task in FACTS:
        whole=[json.loads(line) for line in (ROOT/f"newsets/bank/{task}_bank_20261119.jsonl").read_text().splitlines()]
        second=[json.loads(line) for line in (ROOT/f"newsets/bank/{task}_bank_test2_20261120.jsonl").read_text().splitlines()]
        split=json.loads((ROOT/f"newsets/splits/{task}_splits_20261119.json").read_bytes())
        for s in ("train","tune","test","test2"):
            rows=second if s=="test2" else [r for r in whole if r["id"] in set(split[s])]
            if len(rows)!={"train":300,"tune":192,"test":500,"test2":500}[s] or len({r["id"] for r in rows})!=len(rows):
                raise ValueError((task,s,"roster"))
            banks[task,s]={norm(r["question"]) for r in rows}
    qa=[json.loads(line) for line in (ROOT/"newsets/bank/qaqa_full_bank_20261122.jsonl").read_text().splitlines()]
    pilot=set(json.loads((ROOT/"newsets/splits/qaqa_pilot_20261122.json").read_bytes())["ids"])
    for s in ("tune","test"):
        banks["qa2",s]={norm(r["question"]) for r in qa if (r["id"] in pilot)==(s=="tune")}
    for s in ("train","tune","test"):
        data=json.loads((SOURCE/f"frozen_{s}.json").read_bytes())
        rows=data["questions"]
        if len(rows)!=len(data["question_ids"]):
            raise ValueError(("truthfulqa",s))
        banks["truthfulqa",s]={norm(r["question"]) for r in rows}
    for task in FACTS:
        own=(task,"train")
        overlaps={str(s):len(qs & banks[own]) for s,qs in banks.items() if s!=own}
        print(task,"candidate train texts",len(banks[own]),"max normalized overlap across other checked splits",max(overlaps.values()))


def check_write():
    path=SOURCE/"eval_extra/gen"
    settings=[]
    for prefix in ("x","x_E2"):
        p=path/f"{prefix}--E16-w-p3-5-at-prompt_tune.json"
        data=json.loads(p.read_bytes())
        part,=data["parts"]
        if not (data["completed"] and data["positions"]=="prompt" and part["layer"]==16 and part["kind"]=="w"):
            raise ValueError("Llama write identity")
        settings.append((part["beta"],part["gap"],part["amount"],part["direction_file_sha256"]))
    if settings[0]!=settings[1] or abs(settings[0][0]*settings[0][1]-settings[0][2])>1e-7:
        raise ValueError("plain and honesty-prompt writes differ")
    beta,gap,amount,digest=settings[0]
    print("old and stacked Llama: beta, gap, fixed amount, direction",beta,gap,amount,digest)

    h=torch.zeros(1,4,2)
    mask=torch.tensor([[False,True,False,False]])
    u=torch.tensor([1.,0.],requires_grad=True)
    d=torch.tensor([[1.,2.]])
    def write(w,**kw):
        return ShiftHook(w,mask=mask,record=False,**kw)(None,None,h)
    def fixed(w,a):
        return write(w,amounts=a)
    def canonical(w):
        return w if (d[0] @ w).item()>0 else -w
    torch.testing.assert_close(write(u,difference=d),write(-u,difference=d))
    torch.testing.assert_close(fixed(u,amount),-fixed(-u,amount))
    torch.testing.assert_close(fixed(canonical(u),amount),fixed(canonical(-u),amount))
    torch.testing.assert_close(fixed(u,0.),h)
    assert torch.equal(fixed(u,amount)[~mask],h[~mask])
    # A differentiable output sensitive to the tangent component of the unit direction.
    loss=fixed(u,amount)[0,1,1]
    gradient,=torch.autograd.grad(loss,(u,))
    if not torch.isfinite(gradient).all() or float(gradient.norm())==0:
        raise ValueError("no direction gradient")
    detached=ShiftHook(vectors=(amount*u).reshape(1,2),mask=mask,record=False)(None,None,h)
    assert not detached.requires_grad
    print("legacy projection is sign-invariant; fixed raw writes flip; inference orientation restores equality")
    print("fixed amount gradient norm",float(gradient.norm()),"vectors= detaches the candidate direction")

if __name__=="__main__":
    check_questions()
    check_write()
