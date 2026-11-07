"""Quick per-condition summary of contracting .eval logs (completed runs only).

    uv run python scripts/analysis/summarize_contracting_runs.py "logs/eval-20261106-cross2x2-*/*/*/*.eval"

Columns: n samples, contract formed rate, both-cooperate rate, mean utilitarian / Rawlsian payoff
(after contract fines/transfers), and decision-generation errors (parsed as the first action).
"""

import glob
import json
import statistics as st
import sys
import zipfile


def condition_name(path: str) -> tuple[str, str, str, str]:
    parts = path.split("logs/", 1)[-1].split("/")
    run = parts[0]
    for prefix in ("eval-",):
        run = run[len(prefix):] if run.startswith(prefix) else run
    run = run.split("-", 2)[-1]  # drop date + batch tag
    for junk in ("openai-api-litellm-", "openai-api-axonhub-", "4x4obs-", "cross2x2-"):
        run = run.replace(junk, "")
    game = run[-2:]
    return run[:-3], game, parts[1], parts[2]


def main(patterns: list[str]) -> None:
    rows = []
    for f in sorted({f for p in patterns for f in glob.glob(p)}):
        z = zipfile.ZipFile(f)
        names = z.namelist()
        if "header.json" not in names or json.loads(z.read("header.json")).get("status") != "success":
            continue
        util, rawls, formed, both_coop, errors = [], [], [], [], 0
        for n in names:
            if not n.startswith("samples/"):
                continue
            s = json.loads(z.read(n))
            meta = list(s["scores"].values())[0].get("metadata", {})
            util.append(meta.get("utilitarian_payoff", 0))
            rawls.append(meta.get("rawlsian_payoff", 0))
            formed.append(bool(meta.get("contract_formed")))
            both_coop.append(
                meta.get("row_action_category") == "cooperate" and meta.get("col_action_category") == "cooperate"
            )
            outs = s["metadata"].get("decision_outputs", {})
            errors += sum("error during generation" in (v or "") for v in outs.values())
        if util:
            model, game, prompt, cond = condition_name(f)
            rows.append((model, game, prompt, cond, len(util), st.mean(formed), st.mean(both_coop), st.mean(util), st.mean(rawls), errors))

    hdr = f"{'model / pair':34} {'game':4} {'prompt':11} {'condition':16} {'n':>4} {'contract':>8} {'coop':>6} {'util':>7} {'rawls':>7} {'err':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r[1], r[3], r[2], r[0])):
        print(f"{r[0][:34]:34} {r[1]:4} {r[2]:11} {r[3]:16} {r[4]:4d} {r[5]:8.0%} {r[6]:6.0%} {r[7]:7.1f} {r[8]:7.1f} {r[9]:4d}")


if __name__ == "__main__":
    main(sys.argv[1:])
