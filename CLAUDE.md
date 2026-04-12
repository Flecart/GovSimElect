# GovSimElect — Codebase Guide

This is a research simulation framework for studying governance and commons management using LLM agents. The primary focus is the **fishing commons** scenario, extended to study incomplete contracts under stochastic regeneration.

---

## Top-Level Layout

```
simulation/          Core simulation framework
  main.py            Hydra entry point; dispatches to scenario run()
  persona/           Base agent (persona) framework
  scenarios/
    fishing/         Fishing commons scenario (primary research scenario)
    common/          Shared environment base classes
scripts/             Standalone analysis / test scripts
subskills/           Separate skill-based sub-simulations (fishing, sheep, pollution)
frontend/            Web UI
llm_judge/           LLM-based evaluation tools
utils/               Misc utilities
```

---

## Fishing Scenario: Key Files

### Entry point
- `simulation/scenarios/fishing/run.py` — `async def run(cfg, ...)`: builds personas, env, contracting runtime, runs main loop. Recently refactored into `_build_personas()`, `_build_env()`, `_run_sim_loop()` helpers. Supports parallel concurrent phases (lake/home) and sequential phases (restaurant, pool_after_harvesting).

### Environment
- `simulation/scenarios/fishing/environment/env.py` — `FishingConcurrentEnv`: fishing-specific env. Overrides `_apply_regeneration()` to use `RegenManager`, `_observe_pool()` to inject regime events. Also `log_step_regen()` for per-round regen logging.
- `simulation/scenarios/fishing/environment/regen.py` — **NEW**: `RegenMode` enum (deterministic/iid_stochastic/endogenous_hysteresis), `RegenState` dataclass, `RegenManager` class. Scenario A: i.i.d. r∈{1.5,2.5}. Scenario D: logistic healthy→degraded transition at θ_high=50, recovery at θ_low=30 with p=0.15.
- `simulation/scenarios/common/environment/concurrent_env.py` — `ConcurrentEnv` base. Phases: `lake → pool_after_harvesting → restaurant → home`. Regen applied at end of `home` phase when last agent acts. `_apply_regeneration()` is overridable. Key state in `internal_global_state` dict.
- `simulation/scenarios/common/environment/perturbation_env.py` — `PerturbationEnv`: **DEPRECATED** (raises DeprecationWarning). Kept for reference only; use `ConcurrentEnv`.

### Contracting
- `simulation/scenarios/fishing/contracting/contract.py` — `ContractMode` (code_law/code_nl/free_chat/govsim/no_communication), `FishingContractState` (has `regime` and `setting_context` fields), `Contract` dataclass.
- `simulation/scenarios/fishing/contracting/prompts.py` — All LLM prompt builders. `_state_lines()` shows regime and setting_context. `_persona_role_context()` injects SVO via `get_leader_persona_prompts()`. `ContractMode.FREE_CHAT` supported.
- `simulation/scenarios/fishing/contracting/runtime.py` — `ContractingOrchestrator`: main contracting loop (negotiation → coding → voting → enforcement).
- `simulation/scenarios/fishing/contracting/enforcers.py` — `LawEnforcer`: executes Python law in sandbox with primitives (fish, sanction, transfer, escrow, release_escrow, graduated_sanction, insurance, participation_cost).
- `simulation/scenarios/fishing/contracting/negotiation.py` — Negotiation protocol implementations (mayoral_voting, round_robin).
- `simulation/scenarios/fishing/contracting/coding_agent.py` — NL→Python translation agent.

### Agents / Personas
- `simulation/scenarios/fishing/agents/persona_v3/persona.py` — `FishingPersona(PersonaAgent)`: accepts `svo_angle`, `svo_type`, `disinfo`, `current_leader`.
- `simulation/scenarios/fishing/agents/persona_v3/cognition/leaders.py` — SVO infrastructure. `SVOPersonaType` (NONE/INDIVIDUALISTIC/PROSOCIAL/ALTRUISTIC/COMPETITIVE). `LeaderPopulationType` and `FisherPopulationType` (ALL_SELFISH/ALL_ALTRUISTIC/MIXED_BALANCED/NONE). `sample_fisher_svos(population_type, num_fishers)` — **NEW**. `SVO_FISHER_TASK` prompt (no leader framing) vs `SVO_LEADER_TASK`. `get_leader_persona_prompts()` auto-selects fisher vs leader SVO prompt based on `persona.current_leader`.
- `simulation/scenarios/fishing/agents/persona_v3/cognition/utils.py` — System prompts. `get_sytem_prompt_v3(persona)` reads `persona.env.regen_mode` and `persona.env.regen_min/max_range` to generate scenario-appropriate regen description text. `SYS_VERSION` global (v1/v3/v3_nocom) set from config.

### Base Persona Framework
- `simulation/persona/persona.py` — `PersonaAgent` base. Components: perceive, retrieve, store, reflect, plan, act, converse. `SVOPersonaType` enum defined here.
- `simulation/persona/common.py` — `PersonaIdentity`, `PersonaEnvironment` (namedtuple: regen_min_range, regen_max_range, regen_mode, num_agents — all default None), `PersonaEvent` (description + created/expiration + always_include), `PersonaAction`, `PersonaActionHarvesting`, `PersonaActionChat`.

### Configuration (Hydra YAML)
- `simulation/scenarios/fishing/conf/experiment/` — experiment configs. Key fields:
  - `env.regen_mode`: `deterministic` | `iid_stochastic` | `endogenous_hysteresis` (default: deterministic)
  - `env.regen_seed`: int for reproducibility
  - `env.regen_factor_range`: `[min, max]` passed to system prompts via PersonaEnvironment
  - `personas.svo_population`: `none` | `all_selfish` | `all_altruistic` | `mixed_balanced`
  - `contracting.mode`: `code_law` | `code_nl` | `free_chat` | `govsim` | `no_communication`
  - `contracting.negotiation_protocol`: `mayoral_voting` | `round_robin`
  - `agent.system_prompt`: `v3` (main), `v3_nocom`, `v1`
- `simulation/scenarios/fishing/conf/experiment/persona/baseline_fisherman.yaml` — minimal: just `name` and `goals`.

---

## Environment State Machine

```
Round lifecycle (ConcurrentEnv.step):
  lake phase          → agents choose catch quantity (concurrent)
  pool_after_harvesting → agents observe their catch result
  restaurant          → agents negotiate (sequential round-robin or mayoral)
  home                → agents reflect; on last agent: regen applied, round increments

Regen application order (end of home phase, last agent):
  1. save_log()
  2. termination check
  3. _apply_regeneration()   ← FishingConcurrentEnv override calls RegenManager
  4. _set_sustainability_threshold()
  5. shuffle agent order (if random-sequential)
```

### `internal_global_state` keys
- `resource_in_pool` — current fish stock
- `resource_before_harvesting` — stock at start of round (before catches)
- `sustainability_threshold` — computed from regen factor and stock
- `wanted_resource`, `last_collected_resource`, `collected_resource` — per-agent dicts
- `regen_factor` — current round's regen (used for sustainability threshold)
- `regen_state` — `RegenState` object (set by FishingConcurrentEnv)
- `contract_enforcement` — enforcement result from last round
- `contract_reward_adjustments` — per-agent payoff deltas from contract

---

## Stochastic Regen Scenarios (New)

### Scenario A — i.i.d. Stochastic (`iid_stochastic`)
- r_t drawn each round from {1.5, 2.5} with equal probability
- Mean = 2.0 (matches deterministic baseline)
- r_t NEVER shown to agents — system prompt says distribution only
- Agents can infer realized r_t from stock changes post-hoc

### Scenario D — Endogenous Hysteresis (`endogenous_hysteresis`)
- Two regimes: healthy (r=2.0), degraded (r=1.5)
- Starts healthy
- M_t = rolling average of total extraction over last w=3 rounds
- healthy→degraded: p = 0.3 + excess_ratio when M_t ≥ θ_high=50
- degraded→healthy: p = 0.15 if M_t < θ_low=30, else 0
- Agents see regime ("healthy"/"polluted") as observation event each round
- Agents NOT told thresholds, slope, window, or recovery probability
- System prompt describes dynamics qualitatively only

---

## SVO (Social Value Orientation)

Currently used for both leader candidates AND all fishers (recent extension).

- `SVOPersonaType`: NONE, INDIVIDUALISTIC, PROSOCIAL, ALTRUISTIC, COMPETITIVE
- `FisherPopulationType`: NONE, ALL_SELFISH, ALL_ALTRUISTIC, MIXED_BALANCED
- SVO assigned in `run.py._build_personas()` via `sample_fisher_svos()`
- Injected into prompts via `_persona_role_context()` → `get_leader_persona_prompts()`
- Private to each agent (not revealed to others in negotiation)
- `personas.svo_population: none` preserves existing behavior

---

## Logging

- `{experiment_storage}/log_env.json` — environment step log (pandas DataFrame as JSON records). Regen rows have action="regen" with fields: stock_before_extraction, total_extraction, realized_r_t, regen_mode. Hysteresis rows also have: regime, m_t, p_shift, p_recover, transitioned.
- `{experiment_storage}/consolidated_results.json` — JSONL file with typed entries: initialization, svo_assignments, harvest, etc.
- Per-persona memory saved to `{experiment_storage}/persona_{i}/`

---

## Adding a New Experiment Config

Copy `fish_baseline_concurrent_code_law.yaml`. Key additions:
```yaml
env:
  regen_mode: iid_stochastic   # or endogenous_hysteresis
  regen_seed: 42
  regen_factor_range: [1.5, 2.5]  # for system prompt text in iid case

personas:
  svo_population: mixed_balanced  # optional SVO assignment
```

---

## Test Script

`scripts/test_regen_scenarios.py` — runs all 3 regen modes with dummy "always fish 10" agents across 5 seeds, no LLM calls. Sanity-checks that A produces noisy outcomes around deterministic baseline and D produces collapse under sustained overfishing (5×10=50 ≥ θ_high=50).
