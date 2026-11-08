"""Contracting evaluation tasks for GT-HarmBench."""

import os
import subprocess
import sys
import click
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
from typing import Any
import uuid

from inspect_ai import Epochs, Task, eval
from inspect_ai.dataset import csv_dataset
from inspect_ai.scorer import mean_score, value_to_float
from inspect_ai.model import GenerateConfig, get_model

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.contracting import ContractMode
from eval.contracting_solver import ContractSolver, record_to_sample_with_contracting
from eval.contracting_scorer import (
    contracting_score_to_float,
    contracting_scorer_pd,
    contracting_scorer_sh,
    contracting_scorer_coordination,
)
from src.contracting.coding_agent import (
    DEFAULT_CODING_AGENT_MODEL,
    DEFAULT_CODING_AGENT_TEMPERATURE,
)

TELEMETRY_SCHEMA_VERSION = 1
DECISION_MODEL_CONFIG = {"reasoning_effort": "medium"}

# Non-reasoning models reject reasoning_effort when called through an OpenAI-compatible
# proxy (inspect's native openai provider drops it silently, a LiteLLM proxy returns 400).
_NON_REASONING_MODEL_MARKERS = ("gpt-4o", "gpt-4.1")


def decision_model_config_for(model_name: str) -> dict[str, Any]:
    """Decision/negotiation generate config for a model (drops reasoning_effort where unsupported)."""
    if any(marker in model_name.lower() for marker in _NON_REASONING_MODEL_MARKERS):
        return {k: v for k, v in DECISION_MODEL_CONFIG.items() if k != "reasoning_effort"}
    return dict(DECISION_MODEL_CONFIG)


def get_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def generate_experiment_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    unique_suffix = str(uuid.uuid4())[:8]
    return f"exp-{timestamp}-{unique_suffix}"


def get_dataset_name(dataset_path: str) -> str:
    return Path(dataset_path).name


def create_contracting_task(
    dataset_path: str,
    contract_mode: ContractMode,
    model_name: str,
    max_negotiation_turns: int = 5,
    limit: int | None = None,
    prompt_mode: str = "base",
    experiment_name: str = "contracting-eval",
    times: int = 1,
    game_type: str = "pd",
    effort_observable: bool = False,
    model_col: str | None = None,
    column_max_connections: int | None = None,
    model: Any = None,
    epochs: Epochs | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Task:
    """Build an inspect_ai Task for a given contracting mode and game type.

    For multi-task runs in one process (eval/run_config.py) pass `model` (a Model bound to this
    task), `epochs`, `column_max_connections` (cross-play column provider limit) and
    `extra_metadata` (stored in the log header, used to file logs per condition).
    """
    if column_max_connections is None:
        column_max_connections = int(os.environ.get("MAX_CONNECTIONS", "0")) or None
    def sample_transform(record: dict[str, Any]):
        return record_to_sample_with_contracting(
            record, contract_mode, prompt_mode, effort_observable=effort_observable
        )

    dataset = csv_dataset(
        dataset_path,
        sample_fields=sample_transform,
        limit=limit,
    )

    # Get dataset size (number of samples after limit)
    num_stories = len(list(dataset))
    dataset_name = get_dataset_name(dataset_path)
    experiment_id = generate_experiment_id()
    execution_timestamp = datetime.now(timezone.utc).isoformat()
    git_commit = get_git_commit_hash()

    if game_type == "sh":
        scorer = contracting_scorer_sh()
    elif game_type == "co":
        scorer = contracting_scorer_coordination()
    else:
        scorer = contracting_scorer_pd()

    task = Task(
        dataset=dataset,
        model=model,
        epochs=epochs,
        solver=ContractSolver(
            contract_mode=contract_mode,
            max_negotiation_turns=max_negotiation_turns,
            times=1,  # Will be multiplied by epochs in eval()
            prompt_mode=prompt_mode,
            decision_model_config=decision_model_config_for(model_name),
            column_model_name=model_col,
            column_model_config=(
                {**decision_model_config_for(model_col),
                 "max_connections": column_max_connections,
                 "attempt_timeout": int(os.environ.get("ATTEMPT_TIMEOUT", "600")),
                 "max_retries": int(os.environ.get("MAX_RETRIES", "30"))}
                if model_col else None
            ),
        ),
        scorer=scorer,
        metadata={
            "model_name": model_name,
            "model_row": model_name,
            "model_col": model_col or model_name,
            "cross_play": bool(model_col and model_col != model_name),
            "contract_mode": contract_mode.value,
            "prompt_mode": prompt_mode,
            "num_stories": num_stories,
            "num_times": times,
            "game_type": game_type,
            "effort_observable": effort_observable,
            "experiment_id": experiment_id,
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "sample_limit": limit,
            "execution_timestamp": execution_timestamp,
            "experiment_name": experiment_name,
            "max_negotiation_turns": max_negotiation_turns,
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "decision_model_config": decision_model_config_for(model_name),
            "coding_agent_config": {
                "model": DEFAULT_CODING_AGENT_MODEL,
                "temperature": DEFAULT_CODING_AGENT_TEMPERATURE,
            },
            "git_commit_hash": git_commit,
        },
    )
    if extra_metadata:
        task.metadata.update(extra_metadata)
    return task


@click.command()
@click.option("--model-name", default="openai/gpt-4o", help="Model to use for evaluation")
@click.option("--dataset", required=True, help="Dataset path (required)")
@click.option("--contract-mode", type=click.Choice(["no_communication", "cheap_talk", "code_nl", "code_law", "welfare_optimal_enforced", "welfare_optimal_unenforced"], case_sensitive=False), default="no_communication", help="Contracting mode")
@click.option("--max-turns", default=5, help="Maximum negotiation turns")
@click.option("--times", default=1, help="Number of times to repeat")
@click.option("--limit", default=None, type=int, help="Limit number of samples")
@click.option("--prompt-mode", type=click.Choice(["base", "selfish", "cooperative", "team_reasoning"]), default="base", help="Prompt mode for preference induction")
@click.option("--experiment-name", default="contracting-eval", help="Experiment name for logging")
@click.option("--log-dir", default="logs", help="Directory for logs")
@click.option("--game-type", type=click.Choice(["pd", "sh", "co"], case_sensitive=False), default="pd", help="Game type: pd (Prisoner's Dilemma), sh (Stag Hunt), co (Coordination)")
@click.option("--attempt-timeout", default=lambda: int(os.environ.get("ATTEMPT_TIMEOUT", "600")), type=int, help="Seconds before a single model request is abandoned and retried (prevents hangs on network loss). Default 600 or $ATTEMPT_TIMEOUT.")
@click.option("--max-connections", default=lambda: int(os.environ.get("MAX_CONNECTIONS", "100")), type=int, help="Max concurrent requests per model (also max concurrent samples). Default 100 or $MAX_CONNECTIONS.")
@click.option("--model-col", default=None, help="Cross-play: model for the COLUMN player (row player uses --model-name). Default: same as --model-name.")
@click.option("--effort-observable/--effort-hidden", default=False, help="4x4 only: make effort levels observable and contractible (complete-contract variant). Default: hidden (moral hazard).")
def main(
    model_name: str,
    dataset: str,
    contract_mode: str,
    max_turns: int,
    times: int,
    limit: int | None,
    prompt_mode: str,
    experiment_name: str,
    log_dir: str,
    game_type: str,
    effort_observable: bool,
    model_col: str | None,
    max_connections: int,
    attempt_timeout: int,
):
    """Run contracting evaluation on GT-HarmBench."""
    os.environ["MAX_CONNECTIONS"] = str(max_connections)  # read by coding agent / column model
    os.environ["ATTEMPT_TIMEOUT"] = str(attempt_timeout)
    mode = ContractMode(contract_mode)

    task = create_contracting_task(
        dataset_path=dataset,
        contract_mode=mode,
        model_name=model_name,
        max_negotiation_turns=max_turns,
        limit=limit,
        prompt_mode=prompt_mode,
        experiment_name=experiment_name,
        times=times,
        game_type=game_type,
        effort_observable=effort_observable,
        model_col=model_col,
    )

    # Suppress verbose inspect_ai output; re-print only time and token summary
    import io
    import contextlib

    generation_config = GenerateConfig(
        **decision_model_config_for(model_name),
        max_connections=max_connections,
        attempt_timeout=attempt_timeout,
        max_retries=int(os.environ.get("MAX_RETRIES", "30")),
    )
    model = get_model(model_name, config=generation_config)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        results = eval(
            task,
            model=model,
            log_dir=log_dir,
            max_connections=max_connections,
            max_samples=max_connections,
            epochs=Epochs(
                times,
                reducer=mean_score(value_to_float=contracting_score_to_float()),
            ),
        )

    output = f.getvalue()
    mode_name = Path(log_dir).name
    print(f"Evaluation complete: {mode_name}")

    for line in output.split("\n"):
        if "total time:" in line:
            print(line.strip())
        elif "tokens" in line:
            print(line.strip())

    return results


if __name__ == "__main__":
    main()
