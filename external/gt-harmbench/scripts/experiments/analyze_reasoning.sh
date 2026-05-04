uv run python scripts/analysis/reasoning_analysis.py \
  logs/2026-01-05T14-45-50+01-00_all-strategies_P9emknPZ4bJq6BBXmpAHuY.eval \
  --judge-model openai/gpt-4o-mini \
  --limit 20 \
  --seed 123 \
  --output my_analysis.csv