"""Contracting module for GT-HarmBench.

This module provides support for evaluating LLM behavior in game-theoretic scenarios
with contracting mechanisms (natural language and Python code-law).

Ported from mech-design fishing formal-contract experiments, adapted for 2-player games.
"""

from .contract import (
    Contract,
    ContractMode,
    ContractType,
    EnforcementResult,
    GameContractState,
    NegotiationResult,
)
from .managers import create_contract_manager

__all__ = [
    "Contract",
    "ContractMode",
    "ContractType",
    "EnforcementResult",
    "GameContractState",
    "NegotiationResult",
    "create_contract_manager",
]
