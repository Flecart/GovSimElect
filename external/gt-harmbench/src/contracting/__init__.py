"""Contracting module for GT-HarmBench.

Supports evaluating LLM behavior in game-theoretic scenarios with natural language
and Python code-law contracting mechanisms.
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
