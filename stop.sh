#!/usr/bin/env bash
# Graceful shutdown for Personal Database.
#
# Order matters — the FLM/NPU server must never be stopped (or SIGKILLed)
# while a generation is in flight:
#   1. SIGTERM the app → uvicorn graceful shutdown → lifespan waits for
#      in-flight generations to finish (drain) before the process exits.
#   2. Only once the app is fully down (no clients, no calls) stop FLM.
#
# Usage:
#   ./stop.sh              stop app, then stop FLM
#   KEEP_FLM=1 ./stop.sh   stop app but leave the FLM server running
#   PORT=8765 ./stop.sh    app port (default 8765)

set -uo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8765}"
FLM_PORT="${FLM_PORT:-52625}"
# Must match run.sh's SHUTDOWN_DRAIN_SECONDS plus slack for uvicorn itself.
MAX_WAIT=$(( ${SHUTDOWN_DRAIN_SECONDS:-300} + 60 ))

APP_PAT="uvicorn personal_db.web.app:app"

if ! pgrep -f "$APP_PAT" >/dev/null 2>&1; then
  echo "App is not running."
else
  echo "Stopping app (SIGTERM) — waiting for in-flight generations to drain..."
  pkill -TERM -f "$APP_PAT" 2>/dev/null
  for i in $(seq 1 "$MAX_WAIT"); do
    if ! pgrep -f "$APP_PAT" >/dev/null 2>&1; then
      echo "App stopped cleanly."
      break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
      echo "WARNING: app still running after ${MAX_WAIT}s — sending SIGKILL."
      echo "         If a generation was active, the NPU call is aborted mid-stream."
      pkill -KILL -f "$APP_PAT" 2>/dev/null
    fi
    sleep 1
  done
fi

if [ "${KEEP_FLM:-0}" = "1" ]; then
  echo "KEEP_FLM=1 — leaving FLM running on port ${FLM_PORT}."
  exit 0
fi

if ! curl -sf "http://localhost:${FLM_PORT}/api/tags" >/dev/null 2>&1; then
  echo "FLM is not running."
  exit 0
fi

# App is down, so no client is mid-generation: safe to stop FLM now.
echo "Stopping FLM server (idle — safe)..."
pkill -TERM -f "flm serve" 2>/dev/null || true
echo "Done."
