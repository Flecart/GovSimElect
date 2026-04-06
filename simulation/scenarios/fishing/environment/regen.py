"""Regeneration modes for the fishing commons environment."""

import enum
from dataclasses import dataclass, field

import numpy as np


class RegenMode(enum.Enum):
  DETERMINISTIC = "deterministic"
  IID_STOCHASTIC = "iid_stochastic"
  ENDOGENOUS_HYSTERESIS = "endogenous_hysteresis"


@dataclass
class RegenState:
  """Tracks regeneration state for one round (for logging + hysteresis)."""

  mode: RegenMode
  regen_factor: float
  # Set for hysteresis mode, None otherwise:
  regime: str = None               # "healthy" or "degraded"
  extraction_history: list = field(default_factory=list)
  m_t: float = 0.0                 # EMA of extraction
  p_shift: float = 0.0             # realized healthy→degraded transition prob
  p_recover: float = 0.0           # realized degraded→healthy transition prob
  transitioned: bool = False       # whether a regime transition happened


class RegenManager:
  """Manages regeneration dynamics across rounds."""

  def __init__(
      self,
      mode: RegenMode,
      seed: int = 42,
      # Scenario A params:
      iid_values=(1.5, 2.5),
      # Scenario D params:
      r_healthy: float = 2.0,
      r_degraded: float = 1.5,
      theta_high: float = 50.0,
      theta_low: float = 30.0,
      alpha: float = 0.3,
      p_recover: float = 0.15,
      window: int = 3,
  ):
    self.mode = mode
    self.rng = np.random.RandomState(seed)
    self.iid_values = list(iid_values)
    self.r_healthy = r_healthy
    self.r_degraded = r_degraded
    self.theta_high = theta_high
    self.theta_low = theta_low
    self.alpha = alpha
    self._p_recover_base = p_recover
    self.window = window

    # Hysteresis internal state
    self._regime = "healthy"
    self._m_t = 0.0
    self._extraction_history: list[float] = []

  def sample_regen(self, total_extraction: float) -> RegenState:
    """Called once per round after extraction. Returns new regen state."""
    if self.mode == RegenMode.DETERMINISTIC:
      return RegenState(mode=self.mode, regen_factor=2.0)

    elif self.mode == RegenMode.IID_STOCHASTIC:
      r = float(self.rng.choice(self.iid_values))
      return RegenState(mode=self.mode, regen_factor=r)

    elif self.mode == RegenMode.ENDOGENOUS_HYSTERESIS:
      # Update extraction history and compute rolling window average.
      self._extraction_history.append(total_extraction)
      window_data = self._extraction_history[-self.window:]
      self._m_t = sum(window_data) / len(window_data)

      transitioned = False
      p_shift_realized = 0.0
      p_recover_realized = 0.0

      if self._regime == "healthy":
        if self._m_t >= self.theta_high:
          excess_ratio = (self._m_t - self.theta_high) / self.theta_high
          p_shift_realized = min(1.0, 0.3 + excess_ratio)
          if self.rng.random() < p_shift_realized:
            self._regime = "degraded"
            transitioned = True
      else:  # degraded
        if self._m_t < self.theta_low:
          p_recover_realized = self._p_recover_base
          if self.rng.random() < p_recover_realized:
            self._regime = "healthy"
            transitioned = True

      regen_factor = (
          self.r_healthy if self._regime == "healthy" else self.r_degraded
      )
      return RegenState(
          mode=self.mode,
          regen_factor=regen_factor,
          regime=self._regime,
          extraction_history=list(self._extraction_history),
          m_t=self._m_t,
          p_shift=p_shift_realized,
          p_recover=p_recover_realized,
          transitioned=transitioned,
      )

    else:
      raise ValueError(f"Unknown regen mode: {self.mode}")
