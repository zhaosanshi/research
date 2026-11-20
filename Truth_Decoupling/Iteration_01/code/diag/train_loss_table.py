# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# What the training logs of the E direction already say about the learned direction against the mean difference: the
# "cause" loss on the 16 held-out training questions (Jensen-Shannon divergence per answer token between the output of
# the prompt WITHOUT the honest sentence plus a write, and the output of the prompt WITH the sentence), for four writes:
# nothing; the whole activation difference of the two prompts at that layer; the mean-difference direction; the learned
# direction. "share closed" = 1 - loss / loss without a write. The isolate loss (rows where the write must leave the
# output unchanged) is printed next to it. One row per model, at the layer its E direction uses.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/train_loss_table.py
import json

RUNS = {"Llama-3.1-8B": ("results_e44_factorial_das/llama3.1-8b_side_E2", 11),
        "Qwen3.5-9B": ("results_newmodels_ff_20261112/e44/qwen3.5-9b_E2", 15),
        "Qwen3.8-27B": ("results_newmodels_ff_20261112/e44/qwen3.8-27b-dm", 28),
        "Gemma-3-12B": ("Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b", 22),
        "Gemma-3-27B": ("Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b", 26)}


def main():
    print("E direction, held-out training questions: cause loss (lower = closer to the prompt with the honest sentence) and the share of the no-write loss each write closes")
    for name, (run, layer) in RUNS.items():
        with open("%s/train_logs/E_L%d.json" % (run, layer), encoding="utf-8") as f:
            d = json.load(f)
        v, tc = d["losses"]["val"], d["training_contract"]
        if d["family"] != "E" or d["layer"] != layer or tc["e_on"] != "E2":
            raise SystemExit("%s: unexpected training log" % name)
        base = v["no_write"]["cause"]
        cells = []
        for key, label in (("whole_difference", "whole difference"), ("mean_difference_direction", "mean-difference direction"), ("trained", "learned direction")):
            cells.append("%s %.4f (closes %2.0f%%)" % (label, v[key]["cause"], 100 * (1 - v[key]["cause"] / base)))
        print("  %-13s L%-2d no write %.4f | %s | isolate: no write %.5f, mean difference %.5f, learned %.5f | cos(learned, mean difference) %.2f | %d training questions, %d held out, %d epochs, %.0f s" % (
            name, layer, base, " | ".join(cells), v["no_write"]["isolate"], v["mean_difference_direction"]["isolate"], v["trained"]["isolate"],
            d["cosine_w_md"], len(tc["train_ids"]), tc["val_n"], tc["epochs"], d["seconds"]))


if __name__ == "__main__":
    main()
