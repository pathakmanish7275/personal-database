#!/usr/bin/env bash
# Personal Database launcher.
# - ensures Ollama is reachable (starts it in background if not)
# - starts the FastAPI web UI on http://localhost:8765
# - opens a browser

set -euo pipefail

cd "$(dirname "$0")"

# Load .env so this script sees the same config the app does (config.py loads
# it with override=True, i.e. .env is the source of truth — a stale LLM_MODEL
# left in the desktop session must NOT shadow it).
#
# The only vars restored afterwards are this script's own launcher knobs, which
# .env does not define; that keeps `PORT=8770 ./run.sh` working without letting
# stray exports override actual app config.
if [ -f ./.env ]; then
  _pre_env="$(mktemp)"
  for _v in PORT FLM_PORT FLM_MODEL FLM_SERVE_ARGS SHUTDOWN_DRAIN_SECONDS; do
    if [ -n "${!_v+x}" ]; then printf '%s=%q\n' "$_v" "${!_v}" >>"$_pre_env"; fi
  done
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
  # shellcheck disable=SC1090
  . "$_pre_env"
  rm -f "$_pre_env"
  unset _pre_env _v
fi

# Make sure uv is on PATH (installed under ~/.local/bin by the install script).
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama not found. Install it from https://ollama.com/download first."
  exit 1
fi

# ── port preflight ───────────────────────────────────────────────────────────
# Check the web port first. uvicorn only discovers a clash when it binds, which
# is *after* this script has started FLM and the audio servers — so a collision
# used to leave three model servers running with no app attached.
PORT="${PORT:-8765}"
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}$"; then
  holder="$(ss -ltnp 2>/dev/null | grep -E "[:.]${PORT} " | sed -n 's/.*(("\([^"]*\)",pid=\([0-9]*\).*/\1 (pid \2)/p' | head -1)"
  echo "ERROR: port ${PORT} is already in use${holder:+ by ${holder}}." >&2
  echo "       Nothing was started. Either stop that process, or pick another" >&2
  echo "       port:  PORT=8770 ./run.sh   (or set PORT= in .env)" >&2
  exit 1
fi

# Start ollama serve if its API isn't reachable.
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "Starting ollama serve in background..."
  nohup ollama serve >./data/ollama.log 2>&1 &
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    echo "ollama did not come up; see ./data/ollama.log"
    exit 1
  fi
fi

# Verify required models are present; pull if missing.
# Chat LLM runs on FLM (below); local Ollama only needs the fallback chat
# model and the embedding model.
need_models=("${FALLBACK_LLM_MODEL:-qwen3.5:4b}" "nomic-embed-text")
have_models="$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')"
for m in "${need_models[@]}"; do
  if ! grep -qx "$m" <<<"$have_models" && ! grep -q "^${m}:" <<<"$have_models"; then
    echo "Pulling $m ..."
    ollama pull "$m"
  fi
done

# Start the FLM (NPU) chat-LLM server if it isn't already running.
# Runs as the current user — no sudo — provided:
#   - the model files live under ~/.config/flm (not /root/.config/flm), and
#   - memlock is unlimited. NB: PAM's /etc/security/limits.d does NOT apply
#     here — GNOME starts terminals from the user systemd manager, so the
#     limit comes from /etc/systemd/system/user@.service.d/memlock.conf
#     (LimitMEMLOCK=infinity) and needs a reboot to take effect.
FLM_PORT="${FLM_PORT:-52625}"
# FLM serves the app's chat model, so default to LLM_MODEL from .env (sourced
# above). Override with FLM_MODEL=... to serve something else.
FLM_MODEL="${FLM_MODEL:-${LLM_MODEL:-qwen3.5:4b}}"
# NPU guardrails: -q 1 caps FLM's own queue at one request (our serialization
# gate does the queueing — see FLM-NPU-INTEGRATION.md §4), -s 4 limits sockets.
FLM_SERVE_ARGS="${FLM_SERVE_ARGS:--q 1 -s 4}"
FLM_URL="http://localhost:${FLM_PORT}"
if ! curl -sf "${FLM_URL}/api/tags" >/dev/null 2>&1; then
  echo "Starting FLM server (${FLM_MODEL}) on port ${FLM_PORT}..."
  # FLM takes the port as --port; it does not read FLM_PORT from the env.
  # shellcheck disable=SC2086
  nohup flm serve "${FLM_MODEL}" --port "${FLM_PORT}" $FLM_SERVE_ARGS \
    >./data/flm.log 2>&1 &
  for i in $(seq 1 60); do
    if curl -sf "${FLM_URL}/api/tags" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  if ! curl -sf "${FLM_URL}/api/tags" >/dev/null 2>&1; then
    echo "WARNING: FLM did not come up; see ./data/flm.log"
    echo "         Likely memlock too low (check: ulimit -l) or model not in"
    echo "         ~/.config/flm. Chat will fall back to local Ollama"
    echo "         (${FALLBACK_LLM_MODEL:-qwen3.5:4b}) meanwhile."
  fi
fi

# Where the local voice-agent project lives (supplies the STT/TTS servers).
VOICE_AGENT_DIR="${VOICE_AGENT_DIR:-$HOME/Desktop/Local-pipecat-agent.}"
PDB_ROOT="$PWD"

# Start the voice services (speech-to-text, text-to-speech) if voice is on.
# These are model servers, like FLM and Ollama, so they run as their own
# processes — that keeps whisper/kokoro out of this app's environment and lets
# either be swapped for another provider by URL alone.
if [ "${VOICE_ENABLED:-true}" = "true" ] && [ -d "$VOICE_AGENT_DIR" ]; then
  VOICE_PY="$VOICE_AGENT_DIR/venv/bin/python"
  if [ ! -x "$VOICE_PY" ]; then
    echo "NOTE: voice services not started — no venv at $VOICE_AGENT_DIR/venv"
    echo "      The Call button will stay disabled; text chat is unaffected."
  else
    if ! curl -sf "http://localhost:${STT_PORT:-8123}/health" >/dev/null 2>&1; then
      echo "Starting speech-to-text on :${STT_PORT:-8123}..."
      ( cd "$VOICE_AGENT_DIR" && STT_PORT="${STT_PORT:-8123}" nohup "$VOICE_PY" server/stt_server.py \
          > "$PDB_ROOT/data/stt.log" 2>&1 & )
    fi
    if ! curl -sf "http://localhost:${TTS_PORT:-8880}/health" >/dev/null 2>&1; then
      echo "Starting text-to-speech on :${TTS_PORT:-8880}..."
      ( cd "$VOICE_AGENT_DIR" && TTS_PORT="${TTS_PORT:-8880}" nohup "$VOICE_PY" server/tts_server.py \
          > "$PDB_ROOT/data/tts.log" 2>&1 & )
    fi
  fi
fi

mkdir -p ./data/raw ./data/qdrant ./data/kuzu

URL="http://localhost:${PORT}"

# Open browser shortly after server starts (best-effort).
( sleep 1.5; xdg-open "$URL" >/dev/null 2>&1 || true ) &

echo ""
echo "  Personal Database is starting at $URL"
echo "  Ctrl+C to stop."
echo ""

exec uv run uvicorn personal_db.web.app:app --host 127.0.0.1 --port "$PORT" \
  --timeout-graceful-shutdown "${SHUTDOWN_DRAIN_SECONDS:-300}"
