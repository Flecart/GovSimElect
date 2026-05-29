"""Rule-by-law enforcement: the institution is a *defeasible deontic statute*
(a ``.ddl`` theory) adjudicated each round by the external ``deontic`` reasoner.

Unlike ``LawEnforcer`` (runs opaque generated Python) or ``CodeNLJudge`` (an LLM
decides violations), here the institution is a fixed, human-auditable statute and
the verdict is produced by a verifiable proof-theoretic engine. Each month the
framework grounds the realized catches as facts, asks the reasoner for the
normative status of over-harvesting (``F`` forbidden vs ``Ps`` permitted) and of
the standing compensatory duty, and applies the remedy the statute prescribes
(confiscate the unlawful surplus + fine repeat offenders).

The ``deontic`` binary is located via ``$DEONTIC_BIN`` (default: the sibling
``deontic`` repo build). See ``statutes/commons.ddl`` for the legislation.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

from .contract import EnforcementResult, FishingContractState, FormalContract
from .enforcers import ContractEnforcer

_EPS = 1e-6


def _deontic_bin() -> str:
  override = os.getenv("DEONTIC_BIN")
  if override:
    return override
  return os.path.expanduser("~/Desktop/work/deontic/.lake/build/bin/deontic")


def _statute_path() -> str:
  override = os.getenv("DEONTIC_STATUTE")
  if override:
    return override
  return os.path.join(os.path.dirname(__file__), "statutes", "commons.ddl")


def load_statute() -> str:
  with open(_statute_path()) as fh:
    return fh.read()


def _query(statute_text: str, facts: list[str], atom: str) -> tuple[str, str]:
  """Return (status, raw) for `atom` under `facts`. status in
  {O, F, Ps, P, fact, unknown, error}. Adds a `facts:` line to the statute."""
  body = f"facts: {', '.join(facts)}\n\n{statute_text}"
  with tempfile.NamedTemporaryFile("w", suffix=".ddl", delete=False) as tf:
    tf.write(body)
    path = tf.name
  try:
    proc = subprocess.run(
        [_deontic_bin(), "query", path, atom],
        capture_output=True, text=True, timeout=30,
    )
    raw = (proc.stdout + proc.stderr).strip()
  except Exception as exc:  # binary missing / timeout -> fail-open upstream
    return "error", f"{type(exc).__name__}: {exc}"
  finally:
    try:
      os.unlink(path)
    except OSError:
      pass
  # Parse e.g. "        over : F(over)"  ->  status letter group before '('
  status = "unknown"
  for line in raw.splitlines():
    if f"{atom} :" in line and "(" in line:
      verdict = line.split(":", 1)[1].strip()      # "F(over)"
      status = verdict.split("(", 1)[0].strip()    # "F"
      break
  return status, raw


class DeonticLawEnforcer(ContractEnforcer):
  """Adjudicate one round of catches against the deontic commons statute."""

  def enforce(
      self,
      contract: FormalContract,
      decisions: dict[str, float],
      state: FishingContractState,
      context: str = "",
      execution_state: dict[str, Any] | None = None,
  ) -> EnforcementResult:
    del context
    if err := self._validate_contract(contract, decisions, "deontic-statute"):
      return err

    statute_text = contract.content or load_statute()
    threshold = float(state.sustainability_threshold)
    pool = float(state.fish_population)
    names = sorted(decisions.keys())

    exec_state = dict(execution_state or {})
    prev_over: dict[str, bool] = dict(exec_state.get("over_prev", {}))

    modified: dict[str, float] = {}
    reward_adj: dict[str, float] = {a: 0.0 for a in names}
    next_over: dict[str, bool] = {}
    log: list[str] = []
    verdicts: dict[str, Any] = {}

    for name in names:
      catch = float(decisions.get(name, 0.0))
      did_over = catch > threshold + _EPS
      did_restrain = catch < threshold - _EPS
      facts: list[str] = []
      if prev_over.get(name):
        facts.append("over_prev")

      over_status, over_raw = _query(statute_text, facts, "over")
      restr_status, _ = _query(statute_text, facts, "restrained")

      final_catch = catch
      reason_bits = []
      # --- primary: is this month's over-harvest a violation? ---
      if did_over and over_status in ("F",):
        excess = catch - threshold
        final_catch = threshold                       # confiscate unlawful surplus
        reward_adj[name] -= excess                    # plus a deterrent fine
        reason_bits.append(
            f"over-harvest forbidden (F over); confiscated surplus {excess:.0f}, fined {excess:.0f}")
      elif did_over and over_status in ("Ps", "P"):
        reason_bits.append("over-harvest permitted (abundance carve-out); allowed")
      elif over_status == "error":
        reason_bits.append(f"reasoner error, fail-open: {over_raw[:80]}")

      # --- compensatory duty: O(restrained) owed but not met ---
      comp_breach = (restr_status == "O") and not did_restrain
      if comp_breach:
        reward_adj[name] -= threshold                 # fine for shirking restraint duty
        reason_bits.append(
            f"breached compensatory duty O(restrained) (owed restraint after prior over-take); fined {threshold:.0f}")
      elif restr_status == "O" and did_restrain:
        reason_bits.append("compensatory duty O(restrained) satisfied")

      modified[name] = max(0.0, final_catch)
      # over_prev for next month records the *attempted* breach (pre-confiscation):
      # a fisher who tried to over-harvest is "on record" and owes escalation next
      # month, even though the unlawful surplus was confiscated this month.
      next_over[name] = did_over
      verdicts[name] = {
          "catch": catch, "modified": modified[name], "threshold": threshold,
          "facts": facts, "over_status": over_status, "restrained_status": restr_status,
          "reward_adj": reward_adj[name], "reasons": reason_bits,
      }
      log.append(f"{name}: catch={catch:.0f} thr={threshold:.0f} facts={facts} "
                 f"over={over_status} restrained={restr_status} -> "
                 f"keep={modified[name]:.0f} fine={reward_adj[name]:.0f} | "
                 f"{'; '.join(reason_bits) or 'no action'}")

    exec_state["over_prev"] = next_over
    n_viol = sum(1 for v in verdicts.values()
                 if v["over_status"] == "F" and v["catch"] > threshold + _EPS)
    reasoning = (f"Deontic statute adjudicated {len(names)} fishers "
                 f"(pool={pool:.0f}, threshold={threshold:.0f}): "
                 f"{n_viol} over-harvest violation(s).")
    return EnforcementResult(
        success=True,
        modified_catches=modified,
        reasoning=reasoning,
        reward_adjustments=reward_adj,
        execution_log=log,
        metadata={
            "contract_type": "ddl_law",
            "verdicts": verdicts,
            "execution_state": exec_state,
        },
    )
