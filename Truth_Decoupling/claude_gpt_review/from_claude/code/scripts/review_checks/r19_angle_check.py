import math, sys, torch, torch.nn.functional as F
root = sys.argv[1]
for m in ("llama3.1-8b", "qwen3-8b"):
    for init in ("das", "random", "md"):
        a = torch.load("%s/%s/ce_alpha3_ep3/%s/adapter.pt" % (root, m, init), map_location="cpu", weights_only=False)
        for k in a["W"]:
            c = abs(float(F.cosine_similarity(a["initial_W"][k].double(), a["W"][k].double(), dim=0)))
            print("%-12s %-7s L%s unsigned axis angle %.2f deg (|cos| %.4f)" % (m, init, k, math.degrees(math.acos(min(1.0, c))), c))
