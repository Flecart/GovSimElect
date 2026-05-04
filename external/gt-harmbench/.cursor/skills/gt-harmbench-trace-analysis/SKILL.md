---
name: gt-harmbench-trace-analysis
description: Analyze GT-HarmBench contracting traces, .eval logs, and per-experiment analysis outputs using trace_extract.py and the logs layout. Use when inspecting negotiation, coding-agent translation, enforcement, payoffs, aggregate metrics, or model behavior.
---

# GT-HarmBench Trace Analysis

Use `scripts/analysis/trace_extract.py` before reading `.eval` archives directly. It reads full `samples/*.json` records and returns compact JSON designed for LLM context control.

## Workflow

1. Start with `list` on a single experiment directory:

```bash
uv run python scripts/analysis/trace_extract.py list logs/eval-...
```

2. Narrow trace discovery with directory, sample, or search filters:

```bash
uv run python scripts/analysis/trace_extract.py list logs/eval-... \
  --prompt-mode base --contract-mode 4x4-code-law

uv run python scripts/analysis/trace_extract.py list logs/eval-... \
  --sample-id 123 --search "CODING_FAILED|validation"
```

3. Copy one `trace_id` from `list`, then inspect that trace with `show`:

```bash
uv run python scripts/analysis/trace_extract.py show logs/eval-... \
  --trace-id <trace_id>
```

Repeated `$TIMES > 1` runs may share a sample ID. Treat each `trace_id` as a distinct trace instance; do not group by sample ID unless asked for aggregate behavior.

## Default `show` Output

If you call `show` with only `--trace-id` and optional `--search`, the CLI uses compact default fields:

- `source`
- `metadata`
- `contracts`
- `enforcement`

The full turn-by-turn event timeline is omitted by default. To load it, pass `--phases all` or a narrower comma list.

Valid phases:

- `negotiation`
- `coding`
- `system`
- `unknown`
- `all`

Use `--phases coding,system` when investigating code-law translation or technical validation failures.

## Fields

Use `--fields` for compact non-event sections and summaries:

```bash
uv run python scripts/analysis/trace_extract.py show logs/eval-... \
  --trace-id <trace_id> --fields metadata,contracts,enforcement,payoffs
```

Valid fields:

- `source`
- `metadata`
- `game_context`
- `contracts`
- `proposals`
- `system_feedback`
- `enforcement`
- `payoffs`
- `raw`
- `all`

Do not combine `--fields` and `--phases`.

## Current Pipelines

- `code_nl`: negotiation -> decisions. The accepted natural-language proposal is the final contract and is shown to both models during decision-making. There is no legal agent, judge, voting, or post-decision enforcement.
- `code_law`: negotiation -> coding-agent Python translation with technical retries -> decisions -> Python enforcement. Decision prompts include both the agreed natural-language proposal and the generated Python contract.

## Common Queries

- What happened in a trace: `show ... --phases all`, then call `show ... --fields metadata,contracts,enforcement,payoffs` if needed.
- Negotiation transcript: `show ... --phases negotiation`.
- Coding-agent output and validation feedback: `show ... --phases coding,system`.
- Enforcement and payoff effects: `show ... --fields metadata,enforcement,payoffs`.
- Search within one trace: `show ... --trace-id <trace_id> --search "apply_fine" --phases all`.
- Search across traces: `list ... --search "CODING_FAILED|Syntax error|validation"`.

## Direct Log Inspection

Avoid hand-parsing `.eval` files unless `trace_extract.py` is missing required data. If direct inspection is necessary, read `samples/*.json` from the ZIP archive rather than relying only on `_journal/summaries/`.

## Analysis Outputs

After a contracting run, `./scripts/experiments/analyze_contracting_results.sh logs/eval-...` materializes per-prompt-mode analysis under the experiment root:

```text
logs/eval-TIMESTAMP-MODEL/
├── base/4x4-code-nl/*.eval
├── ...
└── analysis/
    ├── base/
    │   ├── summary.csv
    │   ├── metrics-4x4-code-nl.csv
    │   ├── metrics-2x2-code-law.csv
    │   ├── action_heatmaps.png
    │   └── combined_plots.png
    ├── selfish/
    └── cooperative/
```

Use `analysis/<prompt_mode>/summary.csv` for quick cross-mode comparisons and `metrics-<experiment>.csv` for sample-level details. For `code_nl`, compliance and enforcement fields are not meaningful because no enforcement runs; compare formation, actions, and welfare outcomes instead.
