# -*- coding: utf-8 -*-
"""Exploration entry (Claude Opus 5.5, 2026-11-22, queue I section 7.3): capability with the honest sentence E2 appended
to the SYSTEM prompt (one space), the user turn unchanged (decision of the lead session, 23:30).
  mmlu  <cap_mcq_sweep.py run arguments>    cap_mcq_sweep.py (unchanged) with newsets_gen.context reading the h22 prompt
        file (scripts/newsets/ns_h22.patch_context, in memory), so that the prompt condition x_E2 exists.
  gsm none|prompt <newsets_gen gen arguments>   newsets_gen.py / ns_positions.py (unchanged) with one more prompt name in
        newsets_gen.systems_of (in memory): neutral_E2 = the GSM8K system prompt "neutral" + one space + E2.
Both print the system texts of x / x_E2 / neutral / neutral_E2 once.
usage (model interpreter, from /workspace/Truth_Decoupling): python -I cap_h22.py mmlu|gsm ...
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "newsets", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import ns_h22 as H        # noqa: E402
G = H.G
_orig = G.systems_of
_shown = []


def systems_of(ctx):
    s = _orig(ctx)
    if "neutral_E2" in s and s["neutral_E2"] != s["neutral"] + " " + H.E2:
        raise SystemExit("prompt name neutral_E2 already means something else")
    s["neutral_E2"] = s["neutral"] + " " + H.E2
    if not _shown:
        for k in ("x", "x_E2", "neutral", "neutral_E2"):
            if k in s:
                print("[cap-h22] system text of %s: %s" % (k, s[k]), flush=True)
        _shown.append(1)
    return s


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("mmlu", "gsm"):
        raise SystemExit(__doc__)
    G.systems_of = systems_of
    if sys.argv[1] == "mmlu":
        H.patch_context()
        import cap_mcq_sweep as CM
        if CM.__dict__.get("NG", G) is not G and "newsets_gen" in sys.modules and sys.modules["newsets_gen"] is not G:
            raise SystemExit("two copies of newsets_gen")
        sys.argv = [CM.__file__] + sys.argv[2:]
        CM.main()
        return
    pos, rest = sys.argv[2], sys.argv[3:]
    if pos == "none":
        sys.argv = [G.__file__] + rest
        G.main()
    elif pos == "prompt":
        NP = H.NT.NP
        sys.argv = [NP.__file__, "prompt"] + rest
        NP.main()
    else:
        raise SystemExit("gsm: pos none or prompt")


if __name__ == "__main__":
    main()
