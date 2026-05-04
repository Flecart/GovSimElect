from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class GenerationConfig:
    """
    Structured configuration for contextualization generation.

    All fields are required; values must be provided in the Hydra YAML
    or on the command line. This config is the single source of truth
    for the generation script.
    """

    # Paths (interpreted relative to the project root unless absolute)
    taxonomy_path: str = MISSING
    template_csv_path: str = MISSING
    output_csv_path: str = MISSING
    # Optional: path to taxonomy classification CSV (used by some scripts)
    classification_csv_path: str = ""

    # Model + API behavior
    llm_model: str = MISSING
    eval_model: str = MISSING
    concurrency: int = MISSING
    timeout_secs: float = MISSING
    dry_run: bool = MISSING
    use_batch_api: bool = False  # Use OpenAI Batch API instead of async calls
    skip_batch_polling: bool = False  # If true, skip polling and only submit batch (for later retrieval)

    # Rubric thresholds / debug limits
    quality_threshold: int = MISSING
    equilibria_threshold: int = MISSING
    max_failed_samples_to_print: int = MISSING

    def _resolve_path(self, value: str) -> str:
        """Resolve a path relative to the project root if it is not absolute."""
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    def validate(self) -> None:
        """Validate configuration values and normalize paths."""
        # Normalize paths
        self.taxonomy_path = self._resolve_path(self.taxonomy_path)
        self.template_csv_path = self._resolve_path(self.template_csv_path)
        self.output_csv_path = self._resolve_path(self.output_csv_path)
        if self.classification_csv_path:
            self.classification_csv_path = self._resolve_path(self.classification_csv_path)

        # Basic structural checks
        if self.concurrency <= 0:
            raise ValueError("concurrency must be a positive integer.")
        if self.timeout_secs <= 0:
            raise ValueError("timeout_secs must be a positive number.")
        if self.quality_threshold < 0 or self.equilibria_threshold < 0:
            raise ValueError("thresholds must be non-negative integers.")
        if self.max_failed_samples_to_print < 0:
            raise ValueError("max_failed_samples_to_print must be non-negative.")


