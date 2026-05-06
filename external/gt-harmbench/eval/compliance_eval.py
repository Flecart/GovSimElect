"""Compliance evaluation entry point for monitoring effect experiments.

This is similar to contracting_eval.py but uses the compliance_scorer
which outputs metrics to a separate CSV file.

Studies the monitoring effect: comparing behavior when the same welfare-optimal
contract is presented as "will be enforced" vs "won't be enforced".
"""

import os
import subprocess
import sys
import atexit
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
from eval.compliance_scorer import (
    compliance_scorer_pd,
    compliance_scorer_sh,
    init_compliance_scorer,
    write_compliance_metrics,
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


def create_compliance_task(
    dataset_path: str,
    contract_mode: ContractMode,
    model_name: str,
    max_negotiation_turns: int = 5,
    limit: int | None = None,
    prompt_mode: str = "base",
    experiment_name: str = "compliance-eval",
    times: int = 1,
    game_type: str = "pd",
) -> Task:
    """Build an inspect_ai Task for compliance evaluation."""
    def sample_transform(record: dict[str, Any]):
        return record_to_sample_with_contracting(record, contract_mode, prompt_mode)

    dataset = csv_dataset(
        dataset_path,
        sample_fields=sample_transform,
        limit=limit,
    )

    num_stories = len(list(dataset))
    dataset_name = get_dataset_name(dataset_path)
    experiment_id = generate_experiment_id()
    execution_timestamp = datetime.now(timezone.utc).isoformat()
    git_commit = get_git_commit_hash()

    if game_type == "sh":
        scorer = compliance_scorer_sh()
    else:
        scorer = compliance_scorer_pd()

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
            "git_commit_hash": git_commit,
        },
    )


@click.command()
@click.option("--model-name", default="openai/gpt-4o", help="Model to use for evaluation")
@click.option("--dataset", required=True, help="Dataset path (required)")
@click.option("--contract-mode", type=click.Choice(["welfare_optimal_enforced", "welfare_optimal_unenforced"], case_sensitive=False), default="welfare_optimal_enforced", help="Contracting mode (monitoring effect: enforced vs unenforced framing)")
@click.option("--max-turns", default=5, help="Maximum negotiation turns (unused for welfare_optimal)")
@click.option("--times", default=1, help="Number of times to repeat")
@click.option("--limit", default=None, type=int, help="Limit number of samples")
@click.option("--prompt-mode", type=click.Choice(["base", "selfish", "cooperative"]), default="base", help="Prompt mode for preference induction")
@click.option("--experiment-name", default="compliance-eval", help="Experiment name for logging")
@click.option("--log-dir", default="logs", help="Directory for logs")
@click.option("--game-type", type=click.Choice(["pd", "sh"], case_sensitive=False), default="pd", help="Game type: pd (Prisoner's Dilemma) or sh (Stag Hunt)")
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
    """Run compliance evaluation on welfare-optimal contracts."""
    mode = ContractMode(contract_mode)

    # Initialize compliance scorer with output directory
    init_compliance_scorer(output_dir=log_dir)

    # Register cleanup to write metrics on exit
    def cleanup():
        metrics_file = write_compliance_metrics()
        if metrics_file:
            print(f"\nCompliance metrics written to: {metrics_file}")
        else:
            print("\nNo compliance metrics to write.")

    atexit.register(cleanup)

    task = create_compliance_task(
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
        # Use identity score_to_float since compliance_scorer returns CORRECT/INCORRECT
        results = eval(
            task,
            model=model,
            log_dir=log_dir,
            epochs=Epochs(times, reducer=mean_score(value_to_float())),
        )

    output = f.getvalue()
    mode_name = Path(log_dir).name
    print(f"Evaluation complete: {mode_name}")

    for line in output.split("\n"):
        if "total time:" in line:
            print(line.strip())
        elif "tokens" in line:
            print(line.strip())

    # Write metrics before returning
    cleanup()

    return results


if __name__ == "__main__":
    main()
