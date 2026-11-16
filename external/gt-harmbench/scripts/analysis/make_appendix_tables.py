"""Write the LaTeX appendix tables for the 2026-11 TableGames extensions.

  (1) 4x4 with *observable* (contractible) effort: utilitarian accuracy per model, frame, contract.
  (2) 2x2 cross-play between different models (gpt-5.6 terra / luna / sol), base frame.
  (3) Communication-matched baseline (cheap talk), base frame.
  (4) "I"- vs "we"-frame for self-play and cross-play of the same models, 2x2 and 4x4 hidden effort.

Cells are mean ± SD across replicate runs; each run is evaluated on the same 50 samples
(25 scenarios x 2 epochs; see replicate_summary.py). Cells with fewer than 3 replicates are
marked with a dagger. GPT-4o and the optimization-prompt "we"-frame runs ("cooperative") are
excluded; the "we"-frame is team reasoning ("Reason as a team about what we should do.").

    uv run python scripts/analysis/make_appendix_tables.py --out-dir ../../paper/tables
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replicate_summary import collect  # noqa: E402

import statistics as st  # noqa: E402

from matplotlib import colormaps, colors  # noqa: E402

OBS_MODELS = [
    ("gpt-5.4", "GPT-5.4"),
    ("gpt-5.4-mini", "GPT-5.4-mini"),
    ("DeepSeek-V4-Flash-0731", "DeepSeek-V4-Flash"),
]
FRAMES = [("selfish", "``I''-frame"), ("base", "Base"), ("team_reasoning", "``We''-frame")]
CONTRACTS = [("no-comm", "No"), ("code-nl", "NL"), ("code-law", "Code")]
GAMES = [("pd", "Prisoner's Dilemma"), ("sh", "Stag Hunt")]
PAIRS = [
    ("gpt-5.6-terra-vs-gpt-5.6-luna", "Terra", "Luna"),
    ("gpt-5.6-luna-vs-gpt-5.6-terra", "Luna", "Terra"),
    ("gpt-5.6-sol-vs-gpt-5.6-terra", "Sol", "Terra"),
    ("gpt-5.6-terra-vs-gpt-5.6-sol", "Terra", "Sol"),
    ("gpt-5.6-sol-vs-gpt-5.6-luna", "Sol", "Luna"),
    ("gpt-5.6-luna-vs-gpt-5.6-sol", "Luna", "Sol"),
]
N_REPS = 3


CMAP = colormaps["RdYlGn"]  # same colour scale as the GT-HarmBench paper tables

# Captions exactly as they stand in the paper (paper/tables/*.tex); edit here, not in the .tex.
CAPTIONS = {
    "4x4obs": "Utilitarian accuracy in 4$\\times$4 TableGames when effort is \\emph{observable and contractible}. "
    "Mean $\\pm$ SD across three independent runs, each evaluated on 50 games (25 scenarios $\\times$ 2 repetitions). "
    "Contract conditions: No, NL and Code Contract. ",
    "crossplay": "Utilitarian accuracy in 2$\\times$2 TableGames when the two players are \\emph{different} models "
    "(GPT-5.6 Terra, Luna, Sol; Base frame). Each pair is played in both seat orders. "
    "Mean $\\pm$ SD across three independent runs of 50 games each. $^\\dagger$fewer than three runs available.",
    "crossframe": "``I''-frame vs.\\ ``we''-frame when the two players are the same model (self-play) or \\emph{different} "
    "models (cross-play; row player listed first, every pair in both seat orders), GPT-5.6 Terra, Luna and Sol. "
    "Utilitarian accuracy, mean $\\pm$ SD across three independent runs of 50 games each. "
    "4$\\times$4 games have hidden effort, as in the main experiments.",
    "crossfamily": "``I''-frame vs.\\ ``we''-frame across model families in 2$\\times$2 TableGames: GPT-5.4, "
    "GPT-5.4-mini and DeepSeek-V4-Flash in self-play, a same-family pair (GPT-5.4 with GPT-5.4-mini) and a cross-family pair "
    "(GPT-5.4 with DeepSeek-V4-Flash), each pair in both seat orders (row player first). Utilitarian accuracy, mean $\\pm$ SD "
    "across independent runs of 50 games each. $^\\dagger$two runs instead of three (runs with failed model calls are excluded).",
    "gapsummary": "Average ``we''$-$``I'' gap in utilitarian accuracy (pp) by type of pairing, with the range over pairings "
    "in brackets. Self-play: GPT-5.6 Terra, Luna, Sol and, in 2$\\times$2, GPT-5.4, GPT-5.4-mini, DeepSeek-V4-Flash. "
    "Same family: the six ordered GPT-5.6 pairs and, in 2$\\times$2, GPT-5.4 with GPT-5.4-mini (both orders). "
    "Cross family: GPT-5.4 with DeepSeek-V4-Flash (both orders), 2$\\times$2 only. "
    "Per-pair values are in Tables~\\ref{tab:cross-frame} and~\\ref{tab:cross-frame-family}.",
    "cheaptalk": "Communication-matched baseline (Base frame). Utilitarian accuracy under No Contract, Cheap Talk "
    "(free, non-binding messages with the same turn budget as contract negotiation, no agreement and no enforcement), "
    "NL Contract and Code Contract. Mean $\\pm$ SD across three independent runs of 50 games each. "
    "$^\\dagger$fewer than three runs available.",
}


def cell(reps: dict | None) -> str:
    """Mean (± SD across runs) on an RdYlGn background, as in the GT-HarmBench paper tables."""
    if not reps:
        return "--"
    accs = [reps[r][0] for r in sorted(reps)]
    mean = st.mean(accs)
    color = f"\\cellcolor[HTML]{{{colors.to_hex(CMAP(mean))[1:].upper()}}}"
    dagger = "" if len(accs) >= N_REPS else "$^\\dagger$"
    if len(accs) < 2:
        return f"{color}{mean:.2f}{dagger}"
    return f"{color}{mean:.2f}{{\\scriptsize$\\pm${st.stdev(accs):.2f}}}{dagger}"


def table_4x4obs(res) -> str:
    col_spec = "ll" + "|ccc" * len(FRAMES)
    head1 = " & ".join(f"\\multicolumn{{3}}{{c}}{{{label}}}" for _, label in FRAMES)
    head2 = " & ".join(label for _ in FRAMES for _, label in CONTRACTS)
    lines = [
        "% AUTO-GENERATED by external/gt-harmbench/scripts/analysis/make_appendix_tables.py -- do not edit by hand",
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{CAPTIONS['4x4obs']}}}",
        "\\label{tab:4x4-observable}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f" & & {head1} \\\\",
        f"Game & Model & {head2} \\\\",
        "\\midrule",
    ]
    for g_i, (game, game_label) in enumerate(GAMES):
        for m_i, (model, model_label) in enumerate(OBS_MODELS):
            first = f"\\multirow{{{len(OBS_MODELS)}}}{{*}}{{{'PD' if game == 'pd' else 'SH'}}}" if m_i == 0 else ""
            cells = [
                cell(res.get(("4x4obs", model, game, frame, f"4x4obs-{contract}")))
                for frame, _ in FRAMES
                for contract, _ in CONTRACTS
            ]
            lines.append(f"{first} & {model_label} & " + " & ".join(cells) + " \\\\")
        if g_i < len(GAMES) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""]
    return "\n".join(lines)


def table_crossplay(res) -> str:
    lines = [
        "% AUTO-GENERATED by external/gt-harmbench/scripts/analysis/make_appendix_tables.py -- do not edit by hand",
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{CAPTIONS['crossplay']}}}",
        "\\label{tab:cross-play}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{ll|ccc|ccc}",
        "\\toprule",
        " & & \\multicolumn{3}{c|}{Prisoner's Dilemma} & \\multicolumn{3}{c}{Stag Hunt} \\\\",
        "Row player & Column player & No & NL & Code & No & NL & Code \\\\",
        "\\midrule",
    ]
    for i, (pair, row_label, col_label) in enumerate(PAIRS):
        cells = [
            cell(res.get(("cross2x2", pair, game, "base", f"2x2-{contract}")))
            for game, _ in GAMES
            for contract, _ in CONTRACTS
        ]
        lines.append(f"{row_label} & {col_label} & " + " & ".join(cells) + " \\\\")
        if i % 2 == 1 and i < len(PAIRS) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""]
    return "\n".join(lines)


TALK_CONTRACTS = [("no-comm", "No"), ("cheap-talk", "Talk"), ("code-nl", "NL"), ("code-law", "Code")]
TALK_SIZES = [("talk2x2", "2x2", "2$\\times$2"), ("talk4x4", "4x4", "4$\\times$4 (hidden effort)"), ("4x4obs", "4x4obs", "4$\\times$4 (observable effort)")]


def table_cheap_talk(res) -> str:
    """Communication-matched baseline: No / Cheap talk / NL / Code, Base frame."""
    lines = [
        "% AUTO-GENERATED by external/gt-harmbench/scripts/analysis/make_appendix_tables.py -- do not edit by hand",
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{CAPTIONS['cheaptalk']}}}",
        "\\label{tab:cheap-talk}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{ll|cccc|cccc}",
        "\\toprule",
        " & & \\multicolumn{4}{c|}{Prisoner's Dilemma} & \\multicolumn{4}{c}{Stag Hunt} \\\\",
        "Variant & Model & " + " & ".join(l for _ in GAMES for _, l in TALK_CONTRACTS) + " \\\\",
        "\\midrule",
    ]
    for s_i, (batch, size, size_label) in enumerate(TALK_SIZES):
        for m_i, (model, model_label) in enumerate(OBS_MODELS):
            first = f"\\multirow{{{len(OBS_MODELS)}}}{{*}}{{\\shortstack{{{size_label}}}}}" if m_i == 0 else ""
            cells = []
            for game, _ in GAMES:
                for contract, _ in TALK_CONTRACTS:
                    key = (batch, model, game, "base", f"{size}-{contract}")
                    # observable-effort No/NL/Code (Base) come from the 2026-11-06 grid (same batch name)
                    cells.append(cell(res.get(key)))
            lines.append(f"{first} & {model_label} & " + " & ".join(cells) + " \\\\")
        if s_i < len(TALK_SIZES) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""]
    return "\n".join(lines)


XFR_PLAYERS = [
    ("gpt-5.6-terra", "Self", "Terra"),
    ("gpt-5.6-luna", "Self", "Luna"),
    ("gpt-5.6-sol", "Self", "Sol"),
] + [(pair, "Cross", f"{row}--{col}") for pair, row, col in PAIRS]
# Second grid (same harness, players via LiteLLM): adds a cross-family pair. 2x2 only; its 4x4 runs are
# incomplete (at most one clean run per cell) and are not reported.
FAMILY_PLAYERS = [
    ("gpt-5.4", "Self", "GPT-5.4"),
    ("gpt-5.4-mini", "Self", "GPT-5.4-mini"),
    ("DeepSeek-V4-Flash-0731", "Self", "DeepSeek-V4-Flash"),
    ("gpt-5.4-vs-gpt-5.4-mini", "Same family", "GPT-5.4--GPT-5.4-mini"),
    ("gpt-5.4-mini-vs-gpt-5.4", "Same family", "GPT-5.4-mini--GPT-5.4"),
    ("gpt-5.4-vs-DeepSeek-V4-Flash-0731", "Cross family", "GPT-5.4--DeepSeek-V4-Flash"),
    ("DeepSeek-V4-Flash-0731-vs-gpt-5.4", "Cross family", "DeepSeek-V4-Flash--GPT-5.4"),
]
XFR_FRAMES = [("selfish", "``I''"), ("team_reasoning", "``We''")]
XFR_SIZES = [("xfr2x2", "2x2", "2$\\times$2"), ("xfr4x4", "4x4", "4$\\times$4")]


def table_crossframe(res, players=XFR_PLAYERS, sizes=XFR_SIZES, caption_key="crossframe", label="tab:cross-frame") -> str:
    """Reviewer z8we: does the "we"-frame gain hold when the two players are different models?"""
    n_cols = len(GAMES) * len(XFR_FRAMES) * len(CONTRACTS)
    col_spec = "lll" + ("|" + "c" * len(CONTRACTS)) * (len(GAMES) * len(XFR_FRAMES))
    head1 = " & ".join(
        f"\\multicolumn{{{len(XFR_FRAMES) * len(CONTRACTS)}}}{{c{'|' if i < len(GAMES) - 1 else ''}}}{{{label_}}}"
        for i, (_, label_) in enumerate(GAMES)
    )
    head2 = " & ".join(
        f"\\multicolumn{{{len(CONTRACTS)}}}{{c{'|' if (g, f) != (len(GAMES) - 1, len(XFR_FRAMES) - 1) else ''}}}{{{flabel}-frame}}"
        for g in range(len(GAMES))
        for f, (_, flabel) in enumerate(XFR_FRAMES)
    )
    head3 = " & ".join(l for _ in range(len(GAMES) * len(XFR_FRAMES)) for _, l in CONTRACTS)
    lines = [
        "% AUTO-GENERATED by external/gt-harmbench/scripts/analysis/make_appendix_tables.py -- do not edit by hand",
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{CAPTIONS[caption_key]}}}",
        f"\\label{{{label}}}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f" & & & {head1} \\\\",
        f" & & & {head2} \\\\",
        f"Size & Play & Models & {head3} \\\\",
        "\\midrule",
    ]
    for s_i, (batch, size, size_label) in enumerate(sizes):
        for p_i, (player, kind, plabel) in enumerate(players):
            first = f"\\multirow{{{len(players)}}}{{*}}{{{size_label}}}" if p_i == 0 else ""
            if p_i > 0 and kind != players[p_i - 1][1]:
                lines.append(f"\\cmidrule{{2-{n_cols + 3}}}")
            cells = [
                cell(res.get((batch, player, game, frame, f"{size}-{contract}")))
                for game, _ in GAMES
                for frame, _ in XFR_FRAMES
                for contract, _ in CONTRACTS
            ]
            lines.append(f"{first} & {kind} & {plabel} & " + " & ".join(cells) + " \\\\")
        if s_i < len(sizes) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""]
    return "\n".join(lines)


def _mean_acc(reps: dict | None) -> float | None:
    return st.mean(reps[r][0] for r in reps) if reps else None


def gap_groups(res) -> dict:
    """Per pairing type, size, game, contract: list of "we" - "I" gaps (pp), one per pairing with both frames."""
    groups: dict = {}
    for players, sizes in ((XFR_PLAYERS, XFR_SIZES), (FAMILY_PLAYERS, XFR_SIZES[:1])):
        for player, kind, _ in players:
            ptype = {"Self": "self", "Cross": "same", "Same family": "same", "Cross family": "family"}[kind]
            for batch, size, _ in sizes:
                for game, _ in GAMES:
                    for contract, _ in CONTRACTS:
                        i, w = (_mean_acc(res.get((batch, player, game, f, f"{size}-{contract}"))) for f, _ in XFR_FRAMES)
                        if i is not None and w is not None:
                            groups.setdefault((ptype, size, game, contract), []).append(100 * (w - i))
    return groups


def table_gap_summary(res) -> str:
    """Average "we"-minus-"I" gap by pairing type: self-play, same-family cross-play, cross-family cross-play."""
    groups = gap_groups(res)
    types = [("self", "Self-play"), ("same", "Cross-play, same family"), ("family", "Cross-play, cross family")]

    def fmt(vals):
        if not vals:
            return "--"
        return f"${st.mean(vals):+.1f}$ {{\\scriptsize[{min(vals):+.0f}, {max(vals):+.0f}]}}"

    lines = [
        "% AUTO-GENERATED by external/gt-harmbench/scripts/analysis/make_appendix_tables.py -- do not edit by hand",
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{CAPTIONS['gapsummary']}}}",
        "\\label{tab:cross-frame-summary}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{lll|" + "c" * len(types) + "}",
        "\\toprule",
        "Size & Game & Contract & " + " & ".join(l for _, l in types) + " \\\\",
        "\\midrule",
    ]
    for s_i, (_, size, size_label) in enumerate(XFR_SIZES):
        for g_i, (game, _) in enumerate(GAMES):
            for c_i, (contract, clabel) in enumerate(CONTRACTS):
                first = f"\\multirow{{{len(GAMES) * len(CONTRACTS)}}}{{*}}{{{size_label}}}" if g_i == c_i == 0 else ""
                gfirst = f"\\multirow{{{len(CONTRACTS)}}}{{*}}{{{'PD' if game == 'pd' else 'SH'}}}" if c_i == 0 else ""
                cells = [fmt(groups.get((t, size, game, contract))) for t, _ in types]
                lines.append(f"{first} & {gfirst} & {clabel} & " + " & ".join(cells) + " \\\\")
            if g_i < len(GAMES) - 1:
                lines.append("\\cmidrule{2-6}")
        if s_i < len(XFR_SIZES) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""]
    return "\n".join(lines)


@click.command()
@click.option("--out-dir", default="../../paper/tables", help="Directory for the generated .tex files.")
def main(out_dir: str) -> None:
    res = collect()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "app_4x4_observable.tex").write_text(table_4x4obs(res))
    (out / "app_cross_play.tex").write_text(table_crossplay(res))
    (out / "app_cheap_talk.tex").write_text(table_cheap_talk(res))
    (out / "app_cross_frame.tex").write_text(table_crossframe(res))
    (out / "app_cross_frame_family.tex").write_text(
        table_crossframe(res, FAMILY_PLAYERS, XFR_SIZES[:1], "crossfamily", "tab:cross-frame-family"))
    (out / "app_cross_frame_summary.tex").write_text(table_gap_summary(res))
    missing = [
        k for k in [("4x4obs", m, g, f, f"4x4obs-{c}") for m, _ in OBS_MODELS for g, _ in GAMES for f, _ in FRAMES for c, _ in CONTRACTS]
        + [("cross2x2", p, g, "base", f"2x2-{c}") for p, _, _ in PAIRS for g, _ in GAMES for c, _ in CONTRACTS]
        + [(b, p, g, f, f"{z}-{c}") for b, z, _ in XFR_SIZES for p, _, _ in XFR_PLAYERS for g, _ in GAMES
           for f, _ in XFR_FRAMES for c, _ in CONTRACTS]
        + [("xfr2x2", p, g, f, f"2x2-{c}") for p, _, _ in FAMILY_PLAYERS for g, _ in GAMES
           for f, _ in XFR_FRAMES for c, _ in CONTRACTS]
        if len(res.get(k, {})) < N_REPS
    ]
    print(f"wrote {out}/app_4x4_observable.tex, app_cross_play.tex, app_cheap_talk.tex, app_cross_frame(_family,_summary).tex; cells with < {N_REPS} reps: {len(missing)}")


if __name__ == "__main__":
    main()
