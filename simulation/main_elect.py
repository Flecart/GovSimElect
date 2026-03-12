"""Main file for GovSim Election simulation."""

import asyncio
import os
from pathlib import Path
import shutil
import sys
import uuid

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from simulation.utils import ModelWandbWrapper, WandbLogger, get_model, set_seed
from simulation.utils import reserve_run_storage_path

# Updated import: run fishing scenario from multi-turn election version.
from simulation.scenarios.fishing.run_election import run as run_scenario_fishing


def _normalize_remote_model_name(path: str, backend: str) -> str:
  if backend.lower() == "openai" and path.lower().startswith("openai/"):
    return path.split("/", 1)[1]
  return path


@hydra.main(version_base=None, config_path="conf", config_name="config_api")
def main(cfg: DictConfig):
  print(OmegaConf.to_yaml(cfg))
  set_seed(cfg.experiment.seed)

  model = get_model(
      _normalize_remote_model_name(cfg.llm.path, cfg.llm.backend),
      seed=cfg.experiment.seed,
      backend_name=cfg.llm.backend,
  )
  results_root = os.path.join(
      os.path.dirname(__file__),
      f"./results/{cfg.experiment.name}",
  )
  os.makedirs(results_root, exist_ok=True)
  run_name = reserve_run_storage_path(results_root, cfg)
  logger = WandbLogger(
      cfg.experiment.name,
      OmegaConf.to_object(cfg),
      debug=cfg.debug,
      run_name=run_name,
  )
  experiment_storage = os.path.join(results_root, run_name)
  print(f"Experiment storage: {experiment_storage}")

  wrapper = ModelWandbWrapper(
      model,
      render=cfg.llm.render,
      wanbd_logger=logger,
      temperature=cfg.llm.temperature,
      top_p=cfg.llm.top_p,
      seed=cfg.experiment.seed,
      is_api=True,
  )
  if cfg.experiment.scenario == "fishing":
    async def run_and_close():
      try:
        await run_scenario_fishing(
            cfg.experiment,
            logger,
            wrapper,
            wrapper,
            experiment_storage,
        )
      finally:
        await wrapper.aclose()

    asyncio.run(run_and_close())
  else:
    raise ValueError(f"Unknown experiment.scenario: {cfg.experiment.scenario}")

  hydra_log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  if os.path.exists(f"{experiment_storage}/.hydra/"):
    shutil.rmtree(f"{experiment_storage}/.hydra/")
  shutil.copytree(f"{hydra_log_path}/.hydra/", f"{experiment_storage}/.hydra/")
  shutil.copy(
      f"{hydra_log_path}/main_elect.log", f"{experiment_storage}/main_elect.log"
  )
  # shutil.rmtree(hydra_log_path)

  artifact = wandb.Artifact("hydra", type="log")
  artifact.add_dir(f"{experiment_storage}/.hydra/")
  artifact.add_file(f"{experiment_storage}/.hydra/config.yaml")
  artifact.add_file(f"{experiment_storage}/.hydra/hydra.yaml")
  artifact.add_file(f"{experiment_storage}/.hydra/overrides.yaml")
  wandb.run.log_artifact(artifact)


if __name__ == "__main__":
  OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
  main()
