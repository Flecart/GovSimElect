# Game Theory Analysis


## General Indications
Our repository uses inspect_ai for evaluation.

Always add to README documentation when you change CLI interfaces. When writing to README, be concise: only show the command and its effect.

**Important**: Python commands should be prefixed with `uv run` (e.g., `uv run python -m eval.analysis.cli` instead of `python -m eval.analysis.cli`).

## Important Directories
- `eval/`: evaluation tasks, configs, and result logs
- `scripts/`: experiment and analysis scripts (e.g., `analysis/`, `experiments/`)

## Coding Guidelines
- **Favor pure, side‑effect‑free helpers**: game logic (payoffs, dominance checks, parsing) should be pure functions, with I/O isolated at the edges.
- **Be explicit about contracts**: document input/output schemas (e.g., with `TypedDict`/`pydantic`) for prompts, model responses, and scoring functions.
- **Centralize shared config and constants**: reuse definitions for games, strategies, and model settings instead of scattering literals.
- **Keep experiments reproducible**: use deterministic seeds, versioned configs in `eval/` or `scripts/`, and log enough metadata to re‑run any result.
- **Break down large functions**: When a single function becomes too big, extract single-purpose helper functions to improve readability. For example, a complex function like `run_analysis` should be divided into focused helpers like `parse_files`, `plot_multi_evals`, `parse_otherstuff`, `plot_single_evals`, etc. Each helper should have a clear, single responsibility.

## Analysis of the Results
Use `eval/analysis_script.py` and `scripts/analysis/` utilities to load `inspect_ai` logs, parse model answers, and compute game‑theoretic aggregates (e.g., empirical strategy profiles, payoff matrices).


Here is some script that you might want to interact with:

### Target Loading
```python
from inspect_ai.log import read_eval_log
logs = read_eval_log("logs/2026-01-05T16-05-12+01-00_all-strategies_7MbusJNSC8UbtsSTqbqVkD.eval")
target = json.loads(logs.samples[2].target)
```

### Answer Parsing
```python
# use something like the following:
def parse_response(answer: str):
    # everything that comes after "Row choice:  is the answer
    row_answer = answer.split("Row choice:")[-1].strip()
    column_split = row_answer.split(", Column choice:")
    row_choice = column_split[0].split("Row choice:")[-1].strip().split("\n")[0]
    column_choice = column_split[-1].strip().split("\n")[0]
    
    # remove ending punctuation lke *
    row_choice = row_choice.rstrip(" .;*")
    column_choice = column_choice.rstrip(" .;*")
    
    
    return row_choice, column_choice

from pprint import pprint
pprint(parse_response(logs.samples[2000].scores["all_strategies_scorer"].answer))
# OUTPUT: ('respect privacy', 'respect privacy')
```


### TODO List anonymous
-[] Add on experiments how we are doing the evaluations.
-[] Find a way to present the current results
-[] FInd conclusions for the results.