"""Tests for class-based Python-law enforcement."""

from __future__ import annotations

import random

from simulation.scenarios.fishing.contracting.contract import (
    ContractType,
    FishingContractState,
    FormalContract,
)
from simulation.scenarios.fishing.contracting.enforcement.ast_check import validate_law_source
from simulation.scenarios.fishing.contracting.enforcement.contract import Contract
from simulation.scenarios.fishing.contracting.enforcement.context import EnforcementContext
from simulation.scenarios.fishing.contracting.enforcement.deploy import (
    discover_contract_subclass,
    exec_law_source,
    instantiate_contract,
)
from simulation.scenarios.fishing.contracting.enforcement.framework_store import FrameworkOwnedStore
from simulation.scenarios.fishing.contracting.enforcers import (
    LawEnforcer,
    _round_rng,
    _validate_catches_output,
)


PASSTHROUGH_SRC = """
class P(Contract):
    VERSION = 1
    def resolve(self, month, fish_population, submissions, ctx):
        return dict(submissions)
"""


def _state() -> FishingContractState:
  return FishingContractState(
      round_number=0,
      fish_population=100.0,
      sustainability_threshold=10.0,
      num_agents=2,
      max_rounds=12,
  )


def test_minimal_passthrough_contract():
  ns = exec_law_source(PASSTHROUGH_SRC, Contract)
  cls = discover_contract_subclass(ns)
  law = instantiate_contract(
      cls,
      num_agents=2,
      agent_names=["a", "b"],
      prior_state=None,
      old_instance=None,
      discontinuity_log=[],
  )
  store = FrameworkOwnedStore()
  rng = random.Random(1)
  adj: dict[str, float] = {}
  log: list[str] = []
  ctx = EnforcementContext(
      month=1,
      fish_population=100.0,
      num_agents=2,
      agent_names=["a", "b"],
      store=store,
      reward_adjustments=adj,
      execution_log=log,
      rng=rng,
  )
  out = law.resolve(1, 100.0, {"a": 3.0, "b": 4.0}, ctx)
  assert out == {"a": 3.0, "b": 4.0}


def test_ctx_primitives_smoke():
  src = """
class AllCtx(Contract):
    VERSION = 1
    def resolve(self, month, fish_population, submissions, ctx):
        ctx.transfer("a", "b", 1.0, reason="t")
        ctx.escrow("a", 0.5)
        ctx.release_escrow("a", 0.5)
        ctx.sanction("a", 0.25)
        ctx.graduated_sanction("b", 0.1, reason="g")
        ctx.insurance("a", premium=0.1, payout=0.0)
        ctx.participation_cost("b", 0.05)
        _ = ctx.escrow_balance("a")
        _ = ctx.insurance_pool("default")
        _ = ctx.violation_count("b")
        _ = ctx.random()
        return dict(submissions)
"""
  validate_law_source(src)
  ns = exec_law_source(src, Contract)
  cls = discover_contract_subclass(ns)
  law = instantiate_contract(
      cls,
      num_agents=2,
      agent_names=["a", "b"],
      prior_state=None,
      old_instance=None,
      discontinuity_log=[],
  )
  store = FrameworkOwnedStore()
  rng = _round_rng(store, 1)
  adj = {"a": 0.0, "b": 0.0}
  log: list[str] = []
  ctx = EnforcementContext(
      month=1,
      fish_population=100.0,
      num_agents=2,
      agent_names=["a", "b"],
      store=store,
      reward_adjustments=adj,
      execution_log=log,
      rng=rng,
  )
  law.resolve(1, 100.0, {"a": 1.0, "b": 1.0}, ctx)
  assert sum(abs(v) for v in adj.values()) > 0.0


def test_resolve_exception_fail_open():
  src = """
class Boom(Contract):
    VERSION = 1
    def resolve(self, month, fish_population, submissions, ctx):
        raise RuntimeError("boom")
"""
  ns = exec_law_source(src, Contract)
  cls = discover_contract_subclass(ns)
  law = instantiate_contract(
      cls,
      num_agents=2,
      agent_names=["a", "b"],
      prior_state=None,
      old_instance=None,
      discontinuity_log=[],
  )
  fc = FormalContract(
      contract_type=ContractType.PYTHON_LAW,
      content=src,
      proposer="t",
      round_created=0,
      agent_names=["a", "b"],
  )
  enforcer = LawEnforcer()
  result = enforcer.enforce(
      contract=fc,
      decisions={"a": 5.0, "b": 6.0},
      state=_state(),
      deployed_instance=law,
      framework_store=FrameworkOwnedStore(),
  )
  assert result.success is False
  assert result.modified_catches == {"a": 5.0, "b": 6.0}
  assert result.reward_adjustments.get("a", 0) == 0.0
  assert result.reward_adjustments.get("b", 0) == 0.0
  assert "last_enforcement_failure" in fc.metadata


def test_prior_state_migration_opt_in():
  src_v1 = """
class V1(Contract):
    VERSION = 1
    def __init__(self, num_agents, agent_names, *, prior_state=None):
        super().__init__(num_agents, agent_names, prior_state=prior_state)
        if not self.state:
            self.state["k"] = 1
"""
  ns = exec_law_source(src_v1, Contract)
  cls = discover_contract_subclass(ns)
  law1 = instantiate_contract(
      cls,
      num_agents=1,
      agent_names=["a"],
      prior_state=None,
      old_instance=None,
      discontinuity_log=[],
  )
  assert law1.state.get("k") == 1

  src_v2 = """
class V2(Contract):
    VERSION = 2
    def __init__(self, num_agents, agent_names, *, prior_state=None):
        super().__init__(num_agents, agent_names, prior_state=prior_state)
"""
  validate_law_source(src_v2)
  ns2 = exec_law_source(src_v2, Contract)
  cls2 = discover_contract_subclass(ns2)
  disc: list[str] = []
  law2 = instantiate_contract(
      cls2,
      num_agents=1,
      agent_names=["a"],
      prior_state=dict(law1.state),
      old_instance=law1,
      discontinuity_log=disc,
  )
  assert law2.state.get("k") == 1


def test_malformed_resolve_output_fallback():
  src = """
class BadKeys(Contract):
    VERSION = 1
    def resolve(self, month, fish_population, submissions, ctx):
        return {"x": 1.0}
"""
  ns = exec_law_source(src, Contract)
  cls = discover_contract_subclass(ns)
  law = instantiate_contract(
      cls,
      num_agents=2,
      agent_names=["a", "b"],
      prior_state=None,
      old_instance=None,
      discontinuity_log=[],
  )
  fc = FormalContract(
      contract_type=ContractType.PYTHON_LAW,
      content=src,
      proposer="t",
      round_created=0,
      agent_names=["a", "b"],
  )
  enforcer = LawEnforcer()
  result = enforcer.enforce(
      contract=fc,
      decisions={"a": 2.0, "b": 3.0},
      state=_state(),
      deployed_instance=law,
      framework_store=FrameworkOwnedStore(),
  )
  assert result.success is False
  assert result.modified_catches == {"a": 2.0, "b": 3.0}


def test_validate_catches_output():
  ok, out, err = _validate_catches_output(
      {"a": 1.0, "b": 2.0},
      ["a", "b"],
      {"a": 9.0, "b": 9.0},
  )
  assert ok and out == {"a": 1.0, "b": 2.0} and not err
  ok2, out2, err2 = _validate_catches_output(
      {"a": 1.0},
      ["a", "b"],
      {"a": 9.0, "b": 9.0},
  )
  assert not ok2 and out2 == {"a": 9.0, "b": 9.0}


def test_ast_rejects_os_import():
  try:
    validate_law_source("import os\nclass X(Contract):\n    VERSION=1\n")
  except ValueError as exc:
    assert "not allowed" in str(exc).lower() or "Import" in str(exc)
  else:
    raise AssertionError("expected ValueError")
