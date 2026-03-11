import asyncio
import os
import shutil
import uuid

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from pathfinder import get_model
from simulation.utils import ModelWandbWrapper, WandbLogger, set_seed


def _normalize_remote_model_name(path: str, backend: str) -> str:
    if backend.lower() == "openai" and path.lower().startswith("openai/"):
        return path.split("/", 1)[1]
    return path


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    logger = WandbLogger(cfg.experiment.name, OmegaConf.to_object(cfg), debug=cfg.debug)
    run_name = logger.run_name if logger.run_name else f"{cfg.llm.path}_run_{cfg.llm.iter}_vanilla"
    if "gpt" in cfg.llm.path:
        run_name = os.path.join("gpt", run_name)
    experiment_storage = os.path.join(
        os.path.dirname(__file__),
        f"./results/{cfg.experiment.name}/{run_name}",
    )

    def build_wrapper(llm_cfg):
        model = get_model(
            _normalize_remote_model_name(llm_cfg.path, llm_cfg.backend),
            seed=cfg.seed,
            backend_name=llm_cfg.backend,
        )
        return ModelWandbWrapper(
            model,
            render=llm_cfg.render,
            wanbd_logger=logger,
            temperature=llm_cfg.temperature,
            top_p=llm_cfg.top_p,
            seed=cfg.seed,
            is_api=True,
        )

    if len(cfg.mix_llm) == 0:
        wrapper = build_wrapper(cfg.llm)
        wrappers = [wrapper] * cfg.experiment.personas.num
        wrapper_framework = wrapper
    else:
        if len(cfg.mix_llm) != cfg.experiment.personas.num:
            raise ValueError(
                f"Length of mix_llm should be equal to personas.num: {cfg.experiment.personas.num}"
            )
        unique_configs = {}
        wrappers = []

        for idx, llm_config in enumerate(cfg.mix_llm):
            llm_config = llm_config.llm
            config_key = (
                llm_config.path,
                llm_config.backend,
                llm_config.temperature,
                llm_config.top_p,
            )
            if config_key not in unique_configs:
                unique_configs[config_key] = build_wrapper(llm_config)

            wrappers.append(unique_configs[config_key])

        llm_framework_config = cfg.framework_model
        config_key = (
            llm_framework_config.path,
            llm_framework_config.backend,
            llm_framework_config.temperature,
            llm_framework_config.top_p,
        )
        if config_key not in unique_configs:
            wrapper_framework = build_wrapper(llm_framework_config)
            unique_configs[config_key] = wrapper_framework
        else:
            wrapper_framework = unique_configs[config_key]

    async def close_wrappers():
        seen = set()
        for wrapper in [*wrappers, wrapper_framework]:
            wrapper_id = id(wrapper)
            if wrapper_id in seen:
                continue
            seen.add(wrapper_id)
            await wrapper.aclose()

    async def run_and_close():
        try:
            if cfg.experiment.scenario == "fishing":
                from .scenarios.fishing.run import run as run_scenario_fishing

                await run_scenario_fishing(
                    cfg.experiment,
                    logger,
                    wrappers,
                    wrapper_framework,
                    experiment_storage,
                )
            else:
                raise ValueError(f"Unknown experiment.scenario: {cfg.experiment.scenario}")
        finally:
            await close_wrappers()

    asyncio.run(run_and_close())

    hydra_log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    shutil.copytree(f"{hydra_log_path}/.hydra/", f"{experiment_storage}/.hydra/")
    shutil.copy(f"{hydra_log_path}/main.log", f"{experiment_storage}/main.log")
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
