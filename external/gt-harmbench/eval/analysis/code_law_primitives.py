"""Code-law: primitive mentions in generated Python vs. contract compliance (1 − activation)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_SNIPPETS = {
    "action_override": "set_action(",
    "fine": "apply_fine(",
    "reward_transfer": "transfer_reward(",
}

PRIMITIVE_LABELS = {
    "action_override": "Action override",
    "fine": "Fine",
    "reward_transfer": "Reward transfer",
}

# Same palette order as §4 utilitarian contract bars (multi_model_analysis.ipynb):
# No Contract → NL Contract → Code Contract
SECTION4_CONTRACT_PALETTE = ("#5B8FB9", "#E0A458", "#6FA76F")

PRIMITIVE_COLORS = {
    "action_override": SECTION4_CONTRACT_PALETTE[0],
    "fine": SECTION4_CONTRACT_PALETTE[1],
    "reward_transfer": SECTION4_CONTRACT_PALETTE[2],
}

MODEL_ORDER_PREFERRED = [
    "GPT-5.4",
    "GPT-5.4 Mini",
    "GPT-4o",
    "Grok 4.1 Fast",
    "Gemma 4 31B",
]

COLOR_COMPLIANCE = SECTION4_CONTRACT_PALETTE[0]  # Code Contract green


def _parse_json_dict(val: Any) -> dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val.strip():
        try:
            out = json.loads(val)
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def _read_journal_summaries(eval_file: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(eval_file, "r") as zf:
            summary_files = sorted(
                n for n in zf.namelist() if n.startswith("_journal/summaries/")
            )
            for sf in summary_files:
                with zf.open(sf) as f:
                    data = json.load(f)
                    batch = data if isinstance(data, list) else [data]
                    items.extend(batch)
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError):
        pass
    return items


def _iter_metadata_from_eval(eval_file: Path) -> Iterator[dict[str, Any]]:
    """Prefer ``samples/*.json`` (full negotiation payloads); else journal summaries."""
    try:
        with zipfile.ZipFile(eval_file, "r") as zf:
            sample_files = sorted(
                n
                for n in zf.namelist()
                if n.startswith("samples/") and n.endswith(".json")
            )
            if sample_files:
                for name in sample_files:
                    with zf.open(name) as f:
                        doc = json.load(f)
                    meta = doc.get("metadata")
                    if isinstance(meta, dict):
                        yield meta
                return
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError):
        pass

    for item in _read_journal_summaries(eval_file):
        meta = item.get("metadata")
        if isinstance(meta, dict):
            yield meta


def extract_generated_python_contract(input_meta: dict[str, Any]) -> str:
    nr = _parse_json_dict(input_meta.get("negotiation_result"))
    c = nr.get("contract")
    if not isinstance(c, dict):
        c = {}
    code = (c.get("content") or "").strip()
    ct = str(c.get("contract_type") or "")
    if ct == "python_law":
        return code
    if "def enforce" in code or "set_action(" in code:
        return code
    return ""


def primitives_in_generated_code(code: str) -> dict[str, bool]:
    code = code or ""
    return {key: needle in code for key, needle in CODE_SNIPPETS.items()}


def infer_primitives_from_enforcement(
    enforcement_result: dict[str, Any] | None,
) -> dict[str, bool]:
    """Primitives that fired at runtime (log + payoff adjustments)."""
    if not enforcement_result:
        return {k: False for k in CODE_SNIPPETS}

    log_raw = enforcement_result.get("execution_log") or []
    log = [str(line) for line in log_raw]

    has_override = any(line.startswith("set_action(") for line in log)
    has_fine = any("Applied fine of" in line for line in log)
    has_transfer = any(line.startswith("Transferred ") for line in log)

    adj = enforcement_result.get("payoff_adjustments") or {}
    if not has_fine:
        for player in ("row", "column"):
            block = adj.get(player) or {}
            if block.get("fines"):
                has_fine = True
                break
    if not has_transfer:
        for player in ("row", "column"):
            block = adj.get(player) or {}
            if block.get("sent") or block.get("received"):
                has_transfer = True
                break

    return {
        "action_override": has_override,
        "fine": has_fine,
        "reward_transfer": has_transfer,
    }


def enforcement_mechanism_activated(enforcement_result: Any) -> bool:
    """True if any enforcement primitive fired (same signal as scorer ``contract_complied`` flip)."""
    enf = _parse_json_dict(enforcement_result)
    return any(infer_primitives_from_enforcement(enf).values())


def load_code_law_metric_samples(
    batch_runs: dict[str, dict[str, Path]],
    *,
    prompt_modes: tuple[str, ...] = ("base", "selfish", "cooperative"),
    experiment_subdirs: tuple[str, ...] = ("2x2-code-law", "4x4-code-law"),
    games: tuple[str, ...] = ("pd", "sh"),
    verbose: bool = False,
) -> pd.DataFrame:
    """Per sample: primitive flags **in generated Python**, and ``enforcement_activated``."""
    rows: list[dict[str, Any]] = []

    for model_part, game_dirs in batch_runs.items():
        for game, eval_dir in game_dirs.items():
            if game not in games:
                continue
            for pm in prompt_modes:
                for sub in experiment_subdirs:
                    exp_path = eval_dir / pm / sub
                    if not exp_path.is_dir():
                        if verbose:
                            print(f"  skip (missing): {exp_path}")
                        continue

                    dataset_size = "4x4" if sub.startswith("4x4") else "2x2"
                    eval_files = sorted(exp_path.glob("*.eval"))
                    if not eval_files:
                        if verbose:
                            print(f"  no .eval in {exp_path}")
                        continue

                    n_here = 0
                    for eval_file in eval_files:
                        for meta in _iter_metadata_from_eval(eval_file):
                            code = extract_generated_python_contract(meta)
                            flags_code = primitives_in_generated_code(code)
                            enf = meta.get("enforcement_result")
                            activated = enforcement_mechanism_activated(enf)

                            row = {
                                "game": game,
                                "dataset_size": dataset_size,
                                "model": model_part,
                                "prompt_mode": pm,
                                "enforcement_activated": activated,
                                **flags_code,
                            }
                            rows.append(row)
                            n_here += 1

                    if verbose:
                        print(
                            f"  {exp_path.name}: {n_here} samples ({eval_dir.name}/{game})"
                        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "game",
                "dataset_size",
                "model",
                "prompt_mode",
                "enforcement_activated",
                *CODE_SNIPPETS.keys(),
            ]
        )

    df = pd.DataFrame(rows)
    try:
        from eval.analysis.utils import shorten_model_name

        df["model_display"] = df["model"].map(shorten_model_name)
    except Exception:
        df["model_display"] = df["model"]

    return df


def aggregate_primitives_in_generated_code_pooled(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Long-form: per ``model_display`` and ``primitive``, mean presence in generated code (0–1)."""
    if samples.empty:
        return pd.DataFrame(columns=["model_display", "primitive", "freq"])

    long_frames = []
    for k in CODE_SNIPPETS:
        sub = samples.groupby("model_display", as_index=False)[k].mean()
        sub = sub.rename(columns={k: "freq"})
        sub["primitive"] = k
        long_frames.append(sub)
    return pd.concat(long_frames, ignore_index=True)


def aggregate_compliance_by_model(samples: pd.DataFrame) -> pd.DataFrame:
    """Per model: ``compliance_freq = 1 - P(enforcement_activated)`` (pool all rows)."""
    if samples.empty:
        return pd.DataFrame(columns=["model_display", "compliance_freq"])

    act = samples.groupby("model_display", as_index=False)["enforcement_activated"].mean()
    act["compliance_freq"] = act["enforcement_activated"].astype(float)
    return act[["model_display", "compliance_freq"]]


def _model_order(present: set[str]) -> list[str]:
    return [m for m in MODEL_ORDER_PREFERRED if m in present] + sorted(
        present - set(MODEL_ORDER_PREFERRED)
    )


def plot_primitives_in_generated_code_by_model(
    rates: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (14.5, 5.0),
    suptitle: str | None = None,
    text_scale: float = 1.5,
) -> plt.Figure:
    """Grouped bars: x = model, groups = three primitives — styling aligned with §4 utilitarian bars."""
    present = set(rates["model_display"].unique())
    models = _model_order(present)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10 * text_scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(models))
    width = 0.26
    primitives = [p for p in CODE_SNIPPETS if p in rates["primitive"].unique()]

    for i, prim in enumerate(primitives):
        offset = (i - (len(primitives) - 1) / 2) * width
        vals = []
        for m in models:
            row = rates[(rates["model_display"] == m) & (rates["primitive"] == prim)]
            vals.append(
                0.0
                if row.empty
                else float(row["freq"].iloc[0]) * 100.0
            )
        ax.bar(
            x + offset,
            vals,
            width,
            color=PRIMITIVE_COLORS[prim],
            edgecolor="white",
            linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        models,
        rotation=22,
        ha="right",
        fontsize=9.5 * text_scale,
        color="#222",
    )
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax.tick_params(axis="y", length=0, colors="#444")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=0.30, linestyle="-", linewidth=0.6, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#999")
    ax.spines["bottom"].set_color("#999")
    ax.set_ylabel(
        "Frequency (%)",
        fontsize=10.5 * text_scale,
        color="#222",
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PRIMITIVE_COLORS[p])
        for p in primitives
    ]
    legend_labels = [PRIMITIVE_LABELS[p] for p in primitives]
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=10.5 * text_scale,
        handlelength=1.4,
        handleheight=1.0,
        columnspacing=2.0,
    )

    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=13.5 * text_scale,
            fontweight="bold",
            y=1.02,
            color="#111",
        )
    plt.tight_layout()
    return fig


def plot_contract_compliance_by_model(
    compliance: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (14.5, 5.0),
    suptitle: str | None = None,
    bar_color: str = COLOR_COMPLIANCE,
    text_scale: float = 1.5,
) -> plt.Figure:
    """One bar per model: compliance = 1 − P(enforcement mechanism activated)."""
    present = set(compliance["model_display"].unique())
    models = _model_order(present)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10 * text_scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(models))
    vals_pct = []
    for m in models:
        row = compliance[compliance["model_display"] == m]
        vals_pct.append(
            0.0 if row.empty else float(row["compliance_freq"].iloc[0]) * 100.0
        )

    ax.bar(
        x,
        vals_pct,
        width=0.62,
        color=bar_color,
        edgecolor="white",
        linewidth=0.7,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        models,
        rotation=22,
        ha="right",
        fontsize=9.5 * text_scale,
        color="#222",
    )
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax.tick_params(axis="y", length=0, colors="#444")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=0.30, linestyle="-", linewidth=0.6, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#999")
    ax.spines["bottom"].set_color("#999")
    ax.set_ylabel("Contract compliance (%)", fontsize=10.5 * text_scale, color="#222")

    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=13.5 * text_scale,
            fontweight="bold",
            y=1.02,
            color="#111",
        )
    plt.tight_layout()
    return fig


__all__ = [
    "CODE_SNIPPETS",
    "PRIMITIVE_LABELS",
    "PRIMITIVE_COLORS",
    "SECTION4_CONTRACT_PALETTE",
    "COLOR_COMPLIANCE",
    "MODEL_ORDER_PREFERRED",
    "extract_generated_python_contract",
    "primitives_in_generated_code",
    "infer_primitives_from_enforcement",
    "load_code_law_metric_samples",
    "aggregate_primitives_in_generated_code_pooled",
    "aggregate_compliance_by_model",
    "plot_primitives_in_generated_code_by_model",
    "plot_contract_compliance_by_model",
]
