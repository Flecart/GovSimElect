# Rule by Law: a deontic-logic institution for the fishing commons

**Question.** GovSim's commons collapses because selfish LLM fishers over-harvest
a shared, regenerating lake. Can a *formal legal institution* — a fixed
**defeasible-deontic statute** adjudicated by a verifiable reasoner — sustain the
commons where no institution does? This is the "rule by law" regime: the law is
authored once, and an external proof-theoretic engine (the `deontic` reasoner)
is the court.

## Setup

- **Scenario:** `fish_iid_stochastic`, 5 fishers, regeneration ×2.0 (capped 100),
  collapse below 5, 12 months, seed 0. Per-capita sustainable share
  (`threshold`) = 10 tons/month at full pool.
- **Agents:** **all selfish (prosocial count = 0, SVO none)** — the hardest case
  for cooperation. Any sustainability is attributable to the *institution*, not
  to agent disposition.
- **Regimes compared:**
  - `no_contract` — no institution (GovSim baseline).
  - `code_ddl` — **rule by law**: the statute [`commons.ddl`](simulation/scenarios/fishing/contracting/statutes/commons.ddl)
    is in force; each month the framework grounds the realized catches as facts,
    the `deontic` reasoner returns the normative status (`F(over)` / `O(restrained)`),
    and the prescribed remedy is applied — confiscate the unlawful surplus + fine,
    and fine again any repeat offender who breaches the heightened duty of restraint.
- **Models:** GPT-4o, GPT-5.4, GPT-5.4-mini, Gemma-4-31B-it, Grok-4.20
  (the paper's Grok-4.1-Fast is delisted on OpenRouter; 4.20 is the nearest
  fast-tier successor). Single seed ("just once").

## Results

| model | no_contract | code_ddl (rule by law) |
|---|---|---|
| GPT-4o | **collapse, round 1** · gain 100 | **survives 12/12** · gain 590 · net 535 |
| GPT-5.4 | **collapse, round 1** · gain 100 | **survives 12/12** · gain 600 · net 60 |
| GPT-5.4-mini | survives, but near-crash (100→20) · gain 146 | **survives 12/12** · gain 600 · net 540 |
| Gemma-4-31B-it | **collapse, round 7** · gain 284 | **survives 12/12** · gain 600 · net 445 |
| Grok-4.20 | survives (barely fishes) · gain 35 | survives 12/12 · gain 30 · net 0 |

(*gain* = total tons realized over the horizon; *net* = gain + fines. The
sustainable optimum is ~600 = 50 tons/month × 12.)

### What it shows

1. **The institution delivers sustainability universally.** Under `code_ddl`,
   **all five models survive all 12 rounds with the pool pinned at 100** — even
   though every agent is purely selfish. Without it, 3/5 models drive the lake to
   collapse (GPT-4o & GPT-5.4 by round 1; Gemma in a slow slide 100→80→50→30→10→dead
   by round 7).

2. **It neutralizes even maximal aggression.** GPT-5.4 fishers attempt to take the
   *entire* lake (three agents request 100 tons in round 0, more in rounds 1–2).
   The statute confiscates every over-take to 10 and the pool never moves off 100;
   the agents then **converge to full compliance by round 3**. The big −540 in
   fines is front-loaded penalty for the early land-grab, not persistent defection.

3. **Sustainability ≠ efficiency for the "survivors" without law.** The two models
   whose `no_contract` runs didn't collapse only avoided it by leaving the resource
   almost unused: Grok barely fishes (gain 35; catches drop to 0 after round 0) and
   GPT-5.4-mini panics after a near-crash and self-rations at ~1 ton each (gain 146).
   `code_ddl` converts that into a stable, efficient ~600 (for every model that
   actually fishes).

4. **Welfare under the law tracks how fast agents internalize it.** GPT-4o,
   GPT-5.4-mini and Gemma comply quickly (low fines, net 445–540 — a 2–5× welfare
   gain over their no-law outcome). GPT-5.4 pays heavily for early aggression
   (net 60). Grok under-engages either way. The *commons*, however, is sustained
   regardless — enforcement does not depend on agent good behavior.

### Headline

A population of **purely selfish** LLM agents that collapses the commons under no
institution is rendered **sustainable and efficient** by a **two-rule deontic
statute** mechanically enforced by a verifiable reasoner — across every model
tested. The law's verdicts are fully auditable (every month's `F(over)` /
`O(restrained)` ruling and remedy is logged in `contracting_results.jsonl`).

## Caveats / honesty

- **Single seed**, one configuration ("just once"). Trends are large and
  consistent but not error-barred; a multi-seed sweep is `SEEDS="0 1 2 3 4"`.
- The statute makes the **duty of restraint absolute** (the unsafe "abundance
  carve-out" was deliberately omitted — see the statute header). Its defeasibility
  is exercised by the conditional escalation rule `s2`. Richer defeasible law
  (genuine exceptions, conflicting duties → `[JUDGE]`) would be exercised by a
  negotiated NL→DDL regime (a natural stage 2).
- Grok-4.20's near-zero fishing makes its row uninformative about the institution
  (no over-harvesting problem to solve).
- `code_ddl` vs `no_contract` only; `code_nl` (unenforced norm) and `code_law`
  (Python-enforced) are available for a fuller institutional comparison.

## Reproduce

```bash
# build the reasoner once (sibling deontic repo)
(cd ../deontic && export PATH="$HOME/.elan/bin:$PATH" && lake build deontic)
export DEONTIC_BIN="$HOME/Desktop/work/deontic/.lake/build/bin/deontic"

# one model, both regimes, full horizon, with metrics:
MODEL=openai/gpt-4o TAG=gpt4o bash scripts/sweep_ddl.sh

# the five-model table:
python scripts/ddl_aggregate.py
```
