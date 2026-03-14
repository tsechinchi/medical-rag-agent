#!/usr/bin/env bash
set -u

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <pid> <log_path> [interval_seconds]" >&2
  echo "Example: $0 12345 experiments/ablation_monitor.log 300" >&2
  exit 1
fi

PID="$1"
LOG_PATH="$2"
INTERVAL="${3:-300}"

mkdir -p "$(dirname "$LOG_PATH")"

echo "===== monitor start $(date --iso-8601=seconds) pid=$PID =====" >> "$LOG_PATH"

while kill -0 "$PID" 2>/dev/null; do
  ts="$(date --iso-8601=seconds)"
  stats="$(ps -p "$PID" -o etime=,%cpu=,%mem= | xargs || true)"
  gpu_apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' -v pid="$PID" '$1+0==pid {gsub(/^ +| +$/, "", $2); printf "%s, %s MiB; ", $1, $2}')"
  if [ -z "$gpu_apps" ]; then
    gpu_apps="none"
  fi
  echo "$ts | alive pid=$PID | $stats | gpu_apps=$gpu_apps" >> "$LOG_PATH"
  sleep "$INTERVAL"
done

echo "$(date --iso-8601=seconds) | exited pid=$PID" >> "$LOG_PATH"
