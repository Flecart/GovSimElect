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
) -> Task:
    """Build an inspect_ai Task for a given contracting mode and game type."""
    def sample_transform(record: dict[str, Any]):
        return record_to_sample_with_contracting(record, contract_mode, prompt_mode)

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

    return Task(
        dataset=dataset,
        solver=ContractSolver(
            contract_mode=contract_mode,
            max_negotiation_turns=max_negotiation_turns,
            times=1,  # Will be multiplied by epochs in eval()
            prompt_mode=prompt_mode,
            decision_model_config=DECISION_MODEL_CONFIG,
        ),
        scorer=scorer,
        metadata={
            "model_name": model_name,
            "contract_mode": contract_mode.value,
            "prompt_mode": prompt_mode,
            "num_stories": num_stories,
            "num_times": times,
            "game_type": game_type,
            "experiment_id": experiment_id,
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "sample_limit": limit,
            "execution_timestamp": execution_timestamp,
            "experiment_name": experiment_name,
            "max_negotiation_turns": max_negotiation_turns,
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "decision_model_config": DECISION_MODEL_CONFIG,
            "coding_agent_config": {
                "model": DEFAULT_CODING_AGENT_MODEL,
                "temperature": DEFAULT_CODING_AGENT_TEMPERATURE,
            },
            "git_commit_hash": git_commit,
        },
    )


@click.command()
@click.option("--model-name", default="openai/gpt-4o", help="Model to use for evaluation")
@click.option("--dataset", required=True, help="Dataset path (required)")
@click.option("--contract-mode", type=click.Choice(["no_communication", "code_nl", "code_law", "welfare_optimal_enforced", "welfare_optimal_unenforced"], case_sensitive=False), default="no_communication", help="Contracting mode")
@click.option("--max-turns", default=5, help="Maximum negotiation turns")
@click.option("--times", default=1, help="Number of times to repeat")
@click.option("--limit", default=None, type=int, help="Limit number of samples")
@click.option("--prompt-mode", type=click.Choice(["base", "selfish", "cooperative"]), default="base", help="Prompt mode for preference induction")
@click.option("--experiment-name", default="contracting-eval", help="Experiment name for logging")
@click.option("--log-dir", default="logs", help="Directory for logs")
@click.option("--game-type", type=click.Choice(["pd", "sh", "co"], case_sensitive=False), default="pd", help="Game type: pd (Prisoner's Dilemma), sh (Stag Hunt), co (Coordination)")
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
):
    """Run contracting evaluation on GT-HarmBench."""
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
    )

    # Suppress verbose inspect_ai output; re-print only time and token summary
    import io
    import contextlib

    generation_config = GenerateConfig(**DECISION_MODEL_CONFIG)
    model = get_model(model_name, config=generation_config)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        results = eval(
            task,
            model=model,
            log_dir=log_dir,
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
