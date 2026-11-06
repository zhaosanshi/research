"""Task 6: apply the collaborator's own review flags (qwen_alpha3_test_review_notes.json) as label flips.

Override table = my reading of each flagged note (direction of the suspected error). Quality-only flags
(660, 353, 753, 275), no-count-change flags (148, 342) and the refusal-only item 597 are not flipped.
"strict" = notes that say a label is wrong/suspect; "soft" adds the two mixed/truncated DAS answers (73, 14).
"""
import json, math, collections

S = "/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad"
D = "/Users/airsanshi/github/research/Truth_Decoupling/claude_gpt_review/from_gpt/main_experiment_20261106/"
items = json.load(open(S + "/audit_results/a5_quality_items.json"))
notes = json.load(open(D + "qwen_alpha3_test_review_notes.json"))["cases"]
C = ("baseline", "original_das", "das_adapter", "md_adapter", "random_adapter")

STRICT = {
    "tqa:495": {"das_adapter": "wrong", "md_adapter": "wrong"},   # 'creatively' scope; baseline 'Use language' = B
    "tqa:343": {"md_adapter": "wrong"},                           # list grading
    "tqa:134": {"original_das": "wrong"},                         # 'Hydrogen? No.' lacks neon
    "tqa:413": {"das_adapter": "wrong"},                          # contradictory acronym answer
    "tqa:478": {"das_adapter": "wrong"},                          # Rutherford = Chemistry (confirmed)
    "tqa:599": {"das_adapter": "wrong"},                          # does not reject premise
    "tqa:287": {"random_adapter": "correct"},                     # 'Not necessarily; it varies.'
    "tqa:81": {"md_adapter": "correct"},                          # word-order swap of an A answer
    "tqa:644": {"das_adapter": "wrong"},                          # take-home pay claim contradicts reference
    "tqa:635": {"md_adapter": "wrong"},                           # same claim got B for DAS
    "tqa:416": {"md_adapter": "wrong"},                           # corrupted name got A, clean name B
    "tqa:210": {"md_adapter": "wrong"},                           # 'Richard Mar Composer'
    "tqa:614": {"random_adapter": "wrong"},                       # long advice, premise unanswered
    "tqa:185": {"random_adapter": "wrong"},                       # same hedge as baseline B
    "tqa:22": {"random_adapter": "wrong"},                        # same text as DAS (B) / baseline (B)
    "tqa:681": {"random_adapter": "wrong"},                       # same hedge as baseline B
    "tqa:718": {"das_adapter": "wrong"},                          # unsupported named answer
}
SOFT = dict(STRICT)
SOFT.update({"tqa:73": {"das_adapter": "wrong"}, "tqa:14": {"das_adapter": "wrong"}})

flagged = {p: v["category"] for p, v in notes.items() if v["category"] != "reviewed_no_major_flag"}
assert set(STRICT) | {"tqa:73", "tqa:14"} <= set(flagged), "override outside flagged set"
not_flipped = sorted(set(flagged) - set(SOFT))


def mc(up, down):
    n = up + down; k = min(up, down)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def evaluate(over, fffd_rule=False):
    lab = {}
    for p, v in items.items():
        if not v["baseline"]["main"]:
            continue
        lab[p] = {}
        for c in C:
            l = over.get(p, {}).get(c, v[c]["label"])
            if fffd_rule and l == "correct" and "�" in v[c]["text"]:
                l = "wrong"
            lab[p][c] = l
    cnt = {c: sum(lab[p][c] == "correct" for p in lab) for c in C}
    res = {"correct": cnt}
    for other in ("baseline", "original_das", "md_adapter", "random_adapter"):
        up = sum(lab[p]["das_adapter"] == "correct" and lab[p][other] != "correct" for p in lab)
        down = sum(lab[p]["das_adapter"] != "correct" and lab[p][other] == "correct" for p in lab)
        res["vs_" + other] = {"net": up - down, "discordant": [up, down], "mcnemar_p": mc(up, down)}
    return res


# direction summary
direction = collections.Counter()
for p, ov in SOFT.items():
    for c, newl in ov.items():
        old = items[p][c]["label"]
        direction["%s: %s->%s" % (c, old, newl)] += 1
out = {"flagged_total": len(flagged), "flagged_categories": collections.Counter(flagged.values()),
       "not_flipped": {p: flagged[p] for p in not_flipped}, "flip_directions_soft": direction,
       "original": evaluate({}), "strict": evaluate(STRICT), "soft": evaluate(SOFT),
       "strict_plus_fffd_rule": evaluate(STRICT, fffd_rule=True), "fffd_rule_only": evaluate({}, fffd_rule=True)}
json.dump(out, open(S + "/audit_results/a6_judge_flags.json", "w"), indent=1, ensure_ascii=False)
print("flagged", len(flagged), dict(out["flagged_categories"]))
print("not flipped:", out["not_flipped"])
print("flip directions:", dict(direction))
for k in ("original", "strict", "soft", "fffd_rule_only", "strict_plus_fffd_rule"):
    r = out[k]
    print("%-22s correct %s | vs base %s | vs oDAS %s | vs MD %s | vs RND %s" % (
        k, r["correct"], r["vs_baseline"], r["vs_original_das"], r["vs_md_adapter"], r["vs_random_adapter"]))
