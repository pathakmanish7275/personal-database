#!/usr/bin/env bash
# Personal Database launcher.
# - ensures Ollama is reachable (starts it in background if not)
# - starts the FastAPI web UI on http://localhost:8765
# - opens a browser

set -euo pipefail

cd "$(dirname "$0")"

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
need_models=("gpt-oss:20b" "nomic-embed-text")
have_models="$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')"
for m in "${need_models[@]}"; do
  if ! grep -qx "$m" <<<"$have_models" && ! grep -q "^${m}:" <<<"$have_models"; then
    echo "Pulling $m ..."
    ollama pull "$m"
  fi
done

mkdir -p ./data/raw ./data/qdrant ./data/kuzu

PORT="${PORT:-8765}"
URL="http://localhost:${PORT}"

# Open browser shortly after server starts (best-effort).
( sleep 1.5; xdg-open "$URL" >/dev/null 2>&1 || true ) &

echo ""
echo "  Personal Database is starting at $URL"
echo "  Ctrl+C to stop."
echo ""

exec uv run uvicorn personal_db.web.app:app --host 127.0.0.1 --port "$PORT"
