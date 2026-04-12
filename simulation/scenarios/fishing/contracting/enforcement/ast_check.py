"""Static checks on generated Python-law source (not a full sandbox)."""

from __future__ import annotations

import ast
from typing import FrozenSet

_FORBIDDEN_NAMES: FrozenSet[str] = frozenset({
    "eval",
    "exec",
    "__import__",
    "open",
    "compile",
    "input",
    "breakpoint",
})

_ALLOWED_IMPORT_MODULES: FrozenSet[str] = frozenset({
    "math",
    "itertools",
    "functools",
    "collections",
    "decimal",
    "typing",
    "enum",
    "re",
    "string",
})


def validate_law_source(source: str) -> None:
  """Raise ``ValueError`` with a short message if the law fails static checks."""
  try:
    tree = ast.parse(source)
  except SyntaxError as exc:
    raise ValueError(f"Law source is not valid Python: {exc}") from exc

  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        base = (alias.name or "").split(".")[0]
        if base not in _ALLOWED_IMPORT_MODULES:
          raise ValueError(
              f"Import of {alias.name!r} is not allowed. "
              f"Allowed top-level modules: {sorted(_ALLOWED_IMPORT_MODULES)}.",
          )
    elif isinstance(node, ast.ImportFrom):
      base = (node.module or "").split(".")[0]
      if base and base not in _ALLOWED_IMPORT_MODULES:
        raise ValueError(
            f"From-import from {node.module!r} is not allowed.",
        )
    elif isinstance(node, ast.Call):
      if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_NAMES:
        raise ValueError(f"Call to {node.func.id}() is forbidden in deployed law code.")

  for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
      if isinstance(node.ctx, ast.Load):
        raise ValueError(f"Use of {node.id!r} is forbidden in deployed law code.")
