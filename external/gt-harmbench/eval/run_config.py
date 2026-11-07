"""Run a whole contracting experiment grid from a YAML config in ONE process.

All conditions become inspect tasks executed concurrently on a single event loop
(`inspect_ai.eval(tasks, max_tasks=...)`); concurrency is bounded only by each provider's
`max_connections`. Memory stays at one process (~200 MB + in-flight requests) no matter how
many conditions run at once.

    uv run python -m eval.run_config config/experiments/contracting_20261106.yaml [--dry-run]

Resumable: a condition whose log dir already contains a successful .eval is skipped. Logs are
written to <log_root>/_staging/<name>/ while running (inspect uses one log dir per eval call)
and filed into the per-condition layout when done; interrupted (incomplete) staging logs are
discarded on the next start.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    ("pd", "2x2"): "data/gt-harmbench-pd30-2x2-with-targets.csv",
    ("sh", "2x2"): "data/gt-harmbench-sh30-2x2-with-targets.csv",
    ("pd", "4x4"): "data/gt-harmbench-pd30-4x4-rewritten.csv",
    ("sh", "4x4"): "data/gt-harmbench-sh30-4x4-rewritten.csv",
}
MODE_SLUG = {"no_communication": "no-comm", "code_nl": "code-nl", "code_law": "code-law"}


@dataclass(frozen=True)
class Condition:
    batch: str
    rep: int
    model: str
    model_col: str | None
    game: str
    size: str  # 2x2 | 4x4 | 4x4obs
    contract_mode: str
    prompt_mode: str
    log_dir: str


def _slug(model: str) -> str:
    return re.sub(r"[/:]", "-", re.sub(r"^openrouter/", "", model))


def _col_slug(model: str) -> str:
    return re.sub(r"^openai-api-[^-]+-", "", _slug(model))


def provider_of(model: str) -> str:
    parts = model.split("/")
    return parts[1] if parts[0] == "openai-api" and len(parts) > 2 else parts[0]


def expand(cfg: dict) -> list[Condition]:
    out: list[Condition] = []
    root, name = cfg.get("log_root", "logs"), cfg["name"]
    for batch in cfg["batches"]:
        players: list[tuple[str, str | None]] = [(m, None) for m in batch.get("models", [])]
        for row, col in batch.get("pairs", []):
            players.append((row, col))
            if batch.get("both_orders"):
                players.append((col, row))
        for rep in cfg.get("replicates", [1]):
            tag = f"{batch['name']}{'' if rep == 1 else f'-r{rep}'}"
            for row, col in players:
                who = _slug(row) + (f"-vs-{_col_slug(col)}" if col and col != row else "")
                for game in batch["games"]:
                    for prompt in batch["prompt_modes"]:
                        for mode in batch["contract_modes"]:
                            log_dir = f"{root}/eval-{name}-{tag}-{who}-{game}/{prompt}/{batch['size']}-{MODE_SLUG[mode]}"
                            out.append(Condition(batch["name"], rep, row, col, game, batch["size"], mode, prompt, log_dir))
    return out


def _header(path: str) -> dict | None:
    try:
        z = zipfile.ZipFile(path)
        return json.loads(z.read("header.json")) if "header.json" in z.namelist() else None
    except Exception:
        return None


def is_complete(log_dir: str) -> bool:
    return any((_header(f) or {}).get("status") == "success" for f in glob.glob(f"{log_dir}/*.eval"))


def _runner_alive(lock: Path) -> int | None:
    """PID of a live runner holding this staging dir, if any."""
    try:
        pid = int(lock.read_text().strip())
        os.kill(pid, 0)
        return pid if pid != os.getpid() else None
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def file_staged_logs(staging: Path) -> tuple[int, int]:
    """Move successful staged logs to their condition dir; drop incomplete ones."""
    filed = dropped = 0
    for f in sorted(staging.glob("*.eval")):
        header = _header(str(f))
        target = ((header or {}).get("eval", {}).get("metadata") or {}).get("run_log_dir")
        if header and header.get("status") == "success" and target:
            Path(target).mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), Path(target) / f.name)
            filed += 1
        else:
            f.unlink()
            dropped += 1
    return filed, dropped


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Only list the conditions that would run.")
def main(config_path: str, dry_run: bool) -> None:
    os.chdir(PROJECT_ROOT)
    cfg = yaml.safe_load(Path(config_path).read_text())

    # Environment must be set before importing the contracting modules (coding agent reads it).
    for env_file in cfg.get("env_files", []):
        load_dotenv(env_file, override=True)
    providers = cfg.get("providers", {})
    coding_model = cfg.get("coding_agent_model")
    if coding_model:
        os.environ["CODING_AGENT_MODEL"] = coding_model
        os.environ["MAX_CONNECTIONS"] = str(providers.get(provider_of(coding_model), {}).get("max_connections", 50))
    os.environ["ATTEMPT_TIMEOUT"] = str(cfg.get("attempt_timeout", 600))
    os.environ["MAX_RETRIES"] = str(cfg.get("max_retries", 30))

    from inspect_ai import Epochs, eval as inspect_eval
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_ai.scorer import mean_score

    from eval.contracting_eval import create_contracting_task, decision_model_config_for
    from eval.contracting_scorer import contracting_score_to_float
    from src.contracting import ContractMode

    staging = Path(cfg.get("log_root", "logs")) / "_staging" / cfg["name"]
    staging.mkdir(parents=True, exist_ok=True)
    lock = staging / ".runner.pid"
    other = _runner_alive(lock)

    conditions = expand(cfg)
    todo = [c for c in conditions if not is_complete(c.log_dir)]
    print(f"{len(conditions)} conditions in config, {len(conditions) - len(todo)} already complete, {len(todo)} to run")
    if dry_run:  # strictly read-only: never touches staging or log files
        if other:
            print(f"(runner pid {other} is active; {len(list(staging.glob('*.eval')))} task logs in staging)")
        for c in todo:
            print(f"  {c.log_dir}")
        return
    if other:
        sys.exit(f"another runner (pid {other}) is using {staging}; not starting")
    lock.write_text(str(os.getpid()))

    # Previous runner is gone: file its finished logs, drop interrupted ones.
    filed, dropped = file_staged_logs(staging)
    if filed or dropped:
        print(f"staging sweep: filed {filed} finished logs, dropped {dropped} incomplete")
        todo = [c for c in conditions if not is_complete(c.log_dir)]
    if not todo:
        lock.unlink(missing_ok=True)
        return

    def max_conn(model: str) -> int | None:
        return providers.get(provider_of(model), {}).get("max_connections")

    def player_model(name: str):
        return get_model(
            name,
            config=GenerateConfig(
                **decision_model_config_for(name),
                max_connections=max_conn(name),
                attempt_timeout=int(os.environ["ATTEMPT_TIMEOUT"]),
                max_retries=int(os.environ["MAX_RETRIES"]),
            ),
        )

    samples = cfg.get("samples", {})
    epochs = Epochs(samples.get("epochs", 1), reducer=mean_score(value_to_float=contracting_score_to_float()))
    tasks = []
    for c in todo:
        base_size = "4x4" if c.size.startswith("4x4") else c.size
        tasks.append(
            create_contracting_task(
                dataset_path=DATASETS[(c.game, base_size)],
                contract_mode=ContractMode(c.contract_mode),
                model_name=c.model,
                max_negotiation_turns=cfg.get("max_negotiation_turns", 5),
                limit=samples.get("limit"),
                prompt_mode=c.prompt_mode,
                experiment_name=f"{c.game}-{c.size}-{c.contract_mode}",
                times=samples.get("epochs", 1),
                game_type=c.game,
                effort_observable=c.size == "4x4obs",
                model_col=c.model_col,
                column_max_connections=max_conn(c.model_col) if c.model_col else None,
                model=player_model(c.model),
                epochs=epochs,
                extra_metadata={
                    "run_log_dir": c.log_dir,
                    "config": str(config_path),
                    "batch": c.batch,
                    "replicate": c.rep,
                    "size": c.size,
                },
            )
        )

    run = cfg.get("run", {})
    print(f"running {len(tasks)} tasks in one process (max_tasks={run.get('max_tasks', 32)})", flush=True)
    try:
        inspect_eval(
            tasks,
            log_dir=str(staging),
            max_tasks=run.get("max_tasks", 32),
            max_samples=run.get("max_samples", 50),
            display=run.get("display", "none"),
            fail_on_error=False,
        )
    finally:
        filed, dropped = file_staged_logs(staging)
        lock.unlink(missing_ok=True)
        print(f"filed {filed} finished logs, dropped {dropped} incomplete", flush=True)
    remaining = [c for c in conditions if not is_complete(c.log_dir)]
    print(f"done: {len(conditions) - len(remaining)}/{len(conditions)} conditions complete")
    sys.exit(0 if not remaining else 1)


if __name__ == "__main__":
    main()
