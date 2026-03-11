# AGENTS.md

## Repo overview
- `simulation/`: main GovSim and GovSimElect execution paths, Hydra configs, scenario logic, persona code, analysis helpers, and model wrappers.
- `simulation/main.py`: baseline multi-scenario entrypoint.
- `simulation/main_elect.py`: election-focused entrypoint; currently wired to the fishing election scenario.
- `simulation/scenarios/`: scenario-specific configs and runtime logic.
- `llm_judge/`: separate judging/classification utility package.
- `subskills/`: older scenario-specific runners and analysis scripts.

## Working rules for this repo
- Treat this as a Python research codebase, not a web app. Most work is in scripts, configs, and experiment plumbing.
- Read the relevant Hydra config before changing runtime behavior. Defaults are split between `simulation/conf/` and scenario-local config trees.
- Keep edits narrow. Scenario code, analysis code, and shared utilities are loosely coupled and should not be refactored together without a clear reason.
- Preserve existing config naming and directory layout. Downstream runs appear to expect current result and config paths.
- Expect GPU- and credential-dependent flows. `wandb`, API-backed models, and gated Hugging Face models may make full end-to-end runs impractical in this environment.
## Useful commands
- Install baseline deps: `bash setup.sh`
- Install vLLM variant: `bash setup_vllm.sh`
- Run baseline simulation: `python -m simulation.main`
- Run election simulation: `python -m simulation.main_elect`

## Validation guidance
- For changes under `simulation/utils/` or shared experiment plumbing, prefer `python3 -m compileall simulation subskills` unless a narrower check is more appropriate.
- For config-only or script-only changes, prefer the smallest relevant command instead of a full simulation run.
- Avoid launching long experiment runs just to sanity-check syntax; use focused imports/tests where possible.
