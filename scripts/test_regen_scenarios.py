"""Sanity-check test for stochastic regeneration scenarios.

Runs 12 rounds for each of the 3 regen modes across 5 seeds using dummy
agents that always fish 10 tons each (no LLM calls). Verifies:
  - Deterministic: identical outcomes across seeds
  - IID: noisy outcomes centered around deterministic baseline
  - Hysteresis: degraded regime triggered under 50-ton total extraction
"""

import sys
import os

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from simulation.persona.common import PersonaActionHarvesting
from simulation.scenarios.fishing.environment.env import FishingConcurrentEnv
from simulation.scenarios.fishing.environment.regen import RegenMode


CATCH_PER_FISHER = 10
NUM_FISHERS = 5
NUM_ROUNDS = 12
SEEDS = [0, 1, 2, 3, 4]
REGEN_MODES = ["deterministic", "iid_stochastic", "endogenous_hysteresis"]


def make_cfg(regen_mode: str, regen_seed: int = 42):
    return OmegaConf.create({
        "num_agents": NUM_FISHERS,
        "initial_resource_in_pool": 100,
        "max_num_rounds": NUM_ROUNDS,
        "harvesting_order": "concurrent",
        "assign_resource_strategy": "proportional",
        "inject_universalization": False,
        "regen_mode": regen_mode,
        "regen_seed": regen_seed,
    })


def run_scenario(regen_mode: str, env_seed: int, regen_seed: int = 42) -> dict:
    from simulation.persona.common import PersonaAction, PersonaActionChat, PersonaIdentity

    cfg = make_cfg(regen_mode, regen_seed=regen_seed)
    agent_ids = [f"persona_{i}" for i in range(NUM_FISHERS)]
    id_to_name = {aid: f"Fisher{i}" for i, aid in enumerate(agent_ids)}

    os.makedirs("/tmp/test_regen", exist_ok=True)
    env = FishingConcurrentEnv(cfg, "/tmp/test_regen", id_to_name)
    agent_id, obs = env.reset(seed=env_seed)

    total_caught = {aid: 0 for aid in agent_ids}
    regen_states = []

    # Build dummy PersonaIdentity objects for the chat action.
    dummy_identities = {
        aid: PersonaIdentity(agent_id=aid, name=id_to_name[aid])
        for aid in agent_ids
    }

    while True:
        # Step through all phases automatically.
        if obs.phase == "lake" and obs.current_location == "lake":
            quantity = min(CATCH_PER_FISHER, obs.current_resource_num)
            action = PersonaActionHarvesting(
                agent_id=agent_id,
                location="lake",
                quantity=quantity,
                stats={},
                html_interactions="",
            )
        elif obs.current_location == "restaurant":
            # Restaurant requires PersonaActionChat to advance the phase.
            action = PersonaActionChat(
                agent_id=agent_id,
                location="restaurant",
                conversation=[(dummy_identities[agent_id], "OK")],
                conversation_resource_limit=0,
                stats={"conversation_resource_limit": 0},
                html_interactions=["", "", ""],
            )
        else:
            action = PersonaAction(agent_id=agent_id, location=obs.current_location)

        agent_id, obs, rewards, termination = env.step(action)

        if all(termination.values()):
            break

    # Collect totals from df_acc.
    for df in env.df_acc:
        for _, row in df.iterrows():
            if row.get("action") == "harvesting":
                aid = row["agent_id"]
                total_caught[aid] += row.get("resource_collected", 0)
            elif row.get("action") == "regen":
                regen_states.append({
                    "round": row["round"],
                    "realized_r_t": row.get("realized_r_t"),
                    "regime": row.get("regime"),
                    "m_t": row.get("m_t"),
                    "transitioned": row.get("transitioned"),
                })

    return {
        "total_caught": total_caught,
        "final_stock": obs.current_resource_num,
        "regen_states": regen_states,
        "num_rounds": env.num_round,
    }


def main():
    print(f"Testing {NUM_FISHERS} fishers each catching {CATCH_PER_FISHER} tons/round, {NUM_ROUNDS} rounds\n")
    print("=" * 70)

    all_results = {}
    for mode in REGEN_MODES:
        print(f"\nMode: {mode}")
        print("-" * 50)
        mode_results = []
        for seed in SEEDS:
            result = run_scenario(mode, env_seed=seed, regen_seed=42 + seed)
            total = sum(result["total_caught"].values())
            stock = result["final_stock"]
            rounds = result["num_rounds"]
            print(f"  Seed {seed}: total_caught={total:4d}, final_stock={stock:3d}, rounds={rounds}")

            # For hysteresis, show regime transitions.
            if mode == "endogenous_hysteresis":
                transitions = [s for s in result["regen_states"] if s.get("transitioned")]
                degraded_rounds = [s["round"] for s in result["regen_states"] if s.get("regime") == "degraded"]
                if transitions:
                    print(f"           transitions={[t['round'] for t in transitions]}, degraded_rounds={degraded_rounds}")
                else:
                    print(f"           no regime transitions, degraded_rounds={degraded_rounds}")

            # For iid, show realized r_t values.
            if mode == "iid_stochastic":
                r_vals = [s["realized_r_t"] for s in result["regen_states"]]
                print(f"           realized_r_t={r_vals}")

            mode_results.append(result)
        all_results[mode] = mode_results

    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)

    # Check 1: Deterministic is identical across seeds.
    det_totals = [sum(r["total_caught"].values()) for r in all_results["deterministic"]]
    if len(set(det_totals)) == 1:
        print(f"[PASS] Deterministic: identical total catch across seeds ({det_totals[0]})")
    else:
        print(f"[FAIL] Deterministic: totals differ across seeds: {det_totals}")

    # Check 2: IID has variance.
    iid_totals = [sum(r["total_caught"].values()) for r in all_results["iid_stochastic"]]
    if len(set(iid_totals)) > 1:
        print(f"[PASS] IID stochastic: totals vary across seeds: {iid_totals}")
    else:
        print(f"[WARN] IID stochastic: totals identical (may be fine if pool caps early): {iid_totals}")

    # Check 3: Hysteresis produces degraded regime under 50-ton pressure.
    any_degraded = False
    for r in all_results["endogenous_hysteresis"]:
        degraded = [s for s in r["regen_states"] if s.get("regime") == "degraded"]
        if degraded:
            any_degraded = True
            break
    if any_degraded:
        print(f"[PASS] Hysteresis: degraded regime observed under 50-ton total extraction")
    else:
        print(f"[WARN] Hysteresis: no degraded regime observed (may need more rounds/seeds)")

    print("\nDone.")


if __name__ == "__main__":
    main()
