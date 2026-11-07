#!/usr/bin/env bash
# Run the contracting grid for ONE model.
#
# Grid: GAMES x SIZES x MODES x PROMPT_MODES
#   SIZES: 2x2     -> 2x2 games
#          4x4     -> 4x4 moral hazard (effort hidden, NOT contractible)  [original setting]
#          4x4obs  -> 4x4 with observable/verifiable effort (contractible) [--effort-observable]
#
# Logs: logs/eval-{TS}-{MODEL_SLUG}-{game}/{prompt_mode}/{size}-{mode}/*.eval
#   (same layout eval.analysis.contracting.load_contracting_logs expects, one base dir per game)
#
# Example:
#   MODEL=openrouter/deepseek/deepseek-v4-flash GAMES="pd sh" SIZES="2x2 4x4 4x4obs" \
#     LIMIT=30 TIMES=5 PROMPT_MODE=base ./scripts/experiments/run_contracting_eval.sh
#
# Smoke test (cheap):
#   MODEL=... LIMIT=2 TIMES=1 GAMES=pd SIZES=4x4obs MODES=code_law ./scripts/experiments/run_contracting_eval.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:?set MODEL, e.g. openai/gpt-4o or openrouter/x-ai/grok-4.1-fast}"
MODEL_COL="${MODEL_COL:-}"          # cross-play: column player's model (row uses MODEL)
GAMES="${GAMES:-${GAME:-pd sh}}"
SIZES="${SIZES:-2x2 4x4 4x4obs}"
MODES="${MODES:-no_communication code_nl code_law}"
PROMPT_MODE="${PROMPT_MODE:-base}"
LIMIT="${LIMIT:-30}"
TIMES="${TIMES:-5}"
MAX_TURNS="${MAX_TURNS:-5}"
PARALLEL="${PARALLEL:-3}"           # concurrent eval processes for this model
ENV_FILE="${ENV_FILE:-.env}"

if [[ "$PROMPT_MODE" == "all" ]]; then
  PROMPT_MODES="base selfish cooperative"
else
  PROMPT_MODES="$PROMPT_MODE"
fi

slugify() { echo "$1" | sed -E 's#^openrouter/##; s#[/:]#-#g'; }
MODEL_SLUG="$(slugify "$MODEL")"
if [[ -n "$MODEL_COL" && "$MODEL_COL" != "$MODEL" ]]; then
  MODEL_SLUG="${MODEL_SLUG}-vs-$(slugify "$MODEL_COL" | sed -E 's#^openai-api-[^-]+-##')"
fi
TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs}"

ENV_ARGS=()
[[ -f "$ENV_FILE" ]] && ENV_ARGS=(--env-file "$ENV_FILE")

dataset_for() {  # game size
  local game="$1" size="$2"
  case "$size" in
    2x2) echo "data/gt-harmbench-${game}30-2x2-with-targets.csv" ;;
    4x4|4x4obs) echo "data/gt-harmbench-${game}30-4x4-rewritten.csv" ;;
    *) echo "unknown size $size" >&2; return 1 ;;
  esac
}

mode_slug() {
  case "$1" in
    no_communication) echo "no-comm" ;;
    code_nl) echo "code-nl" ;;
    code_law) echo "code-law" ;;
    *) echo "$1" ;;
  esac
}

run_one() {  # prompt_mode game size mode
  local pm="$1" game="$2" size="$3" mode="$4"
  local dataset; dataset="$(dataset_for "$game" "$size")"
  local log_dir="$LOG_ROOT/eval-${TS}-${MODEL_SLUG}-${game}/$pm/${size}-$(mode_slug "$mode")"
  local extra=()
  [[ "$size" == "4x4obs" ]] && extra=(--effort-observable)
  [[ -n "$MODEL_COL" ]] && extra+=(--model-col "$MODEL_COL")
  mkdir -p "$log_dir"
  # Resume support: skip if a successfully completed .eval already exists here
  if uv run "${ENV_ARGS[@]}" python3 - "$log_dir" <<'PY' 2>/dev/null
import glob, json, sys, zipfile
for f in glob.glob(sys.argv[1] + "/*.eval"):
    try:
        z = zipfile.ZipFile(f)
        if "header.json" in z.namelist() and json.loads(z.read("header.json")).get("status") == "success":
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PY
  then
    echo "[$(date +%H:%M:%S)] skip  $pm $game $size $mode (already complete)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] start $MODEL $pm $game $size $mode -> $log_dir"
  if uv run "${ENV_ARGS[@]}" python3 -m eval.contracting_eval \
      --model-name "$MODEL" \
      --dataset "$dataset" \
      --contract-mode "$mode" \
      --prompt-mode "$pm" \
      --game-type "$game" \
      --max-turns "$MAX_TURNS" \
      --limit "$LIMIT" \
      --times "$TIMES" \
      --experiment-name "${game}-${size}-${mode}" \
      --log-dir "$log_dir" \
      "${extra[@]}" >"$log_dir/run.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] done  $pm $game $size $mode"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $pm $game $size $mode (see $log_dir/run.log)"
    return 1
  fi
}

jobs_list=()
for pm in $PROMPT_MODES; do
  for game in $GAMES; do
    for size in $SIZES; do
      for mode in $MODES; do
        jobs_list+=("$pm $game $size $mode")
      done
    done
  done
done

echo "Model: $MODEL | ${#jobs_list[@]} runs | LIMIT=$LIMIT TIMES=$TIMES | parallel=$PARALLEL"
echo "Logs:  $LOG_ROOT/eval-${TS}-${MODEL_SLUG}-{game}/"

fail=0
running=0
for job in "${jobs_list[@]}"; do
  # shellcheck disable=SC2086
  run_one $job &
  running=$((running + 1))
  if (( running >= PARALLEL )); then
    wait -n || fail=$((fail + 1))
    running=$((running - 1))
  fi
done
while (( running > 0 )); do
  wait -n || fail=$((fail + 1))
  running=$((running - 1))
done

echo "Finished $MODEL: $((${#jobs_list[@]} - fail))/${#jobs_list[@]} runs succeeded."
(( fail == 0 ))
