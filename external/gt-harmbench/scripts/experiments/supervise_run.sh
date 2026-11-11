#!/usr/bin/env bash
# Keep an eval.run_config grid running until every condition is complete.
#
#   scripts/experiments/supervise_run.sh config/experiments/crossframe_20261108.yaml
#
# Every CHECK_SECS it checks the runner for this config:
#   * not running and the log does not end with "done: N/N" -> relaunch it
#   * resident memory above LIMIT_KB                         -> stop it with SIGINT, relaunch it
# Restarts are safe: run_config files finished conditions and resumes from them.
# At most MAX_RESTARTS restarts. Starts the runner itself if none is running.
# The runner log is logs/_runner/<config name>.log. Sleep is inhibited while the runner works.
#
# Env overrides: MAX_RESTARTS (6), LIMIT_KB (4000000), CHECK_SECS (120).
set -u
CFG=${1:?usage: supervise_run.sh <config.yaml>}
cd "$(dirname "$0")/../.." || exit 1
[ -f "$CFG" ] || { echo "config not found: $CFG"; exit 1; }
NAME=$(basename "$CFG" .yaml)
LOG=logs/_runner/$NAME.log
MAX_RESTARTS=${MAX_RESTARTS:-6}
LIMIT_KB=${LIMIT_KB:-4000000}
CHECK_SECS=${CHECK_SECS:-120}
mkdir -p logs/_runner

runner_pid() {
  # the interpreter itself (not the systemd-inhibit / uv wrappers)
  ps -eo pid,args | awk -v cfg="-m eval.run_config $CFG" '$2 ~ /python[0-9.]*$/ && index($0, cfg) {print $1; exit}'
}
launch() {
  echo "=== supervisor (re)start $(date)" >> "$LOG"
  nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --who=gt-harmbench --why="$NAME runs" \
    uv run python -m eval.run_config "$CFG" >> "$LOG" 2>&1 &
  sleep 30
}
stop() {
  local p=$1; kill -INT "$p"
  for _ in $(seq 1 90); do kill -0 "$p" 2>/dev/null || return; sleep 2; done
  kill -9 "$p"
}
all_done() {  # last summary line reads "done: N/N conditions complete"
  tail -5 "$LOG" 2>/dev/null | awk '$1 == "done:" {split($2, a, "/"); ok = (a[1] == a[2])} END {exit !ok}'
}

[ -z "$(runner_pid)" ] && { echo "no runner for $CFG, starting one $(date)"; launch; }
restarts=0
while true; do
  sleep "$CHECK_SECS"
  p=$(runner_pid)
  if [ -z "$p" ]; then
    all_done && { echo "all conditions complete $(date)"; exit 0; }
    [ "$restarts" -ge "$MAX_RESTARTS" ] && { echo "runner exited, restart budget used $(date)"; exit 1; }
    echo "runner exited early, relaunching $(date)"; restarts=$((restarts+1)); launch; continue
  fi
  rss=$(ps -o rss= -p "$p" | tr -d ' ')
  if [ "${rss:-0}" -gt "$LIMIT_KB" ]; then
    [ "$restarts" -ge "$MAX_RESTARTS" ] && { echo "RSS high, restart budget used $(date)"; exit 1; }
    echo "RSS $rss KB > limit, restarting $(date)"; stop "$p"; restarts=$((restarts+1)); launch
  fi
done
