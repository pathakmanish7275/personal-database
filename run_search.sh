#!/usr/bin/env bash
# Lightweight search-only API — no web UI, no chat, no sessions.
# Runs on port 8766 by default; override with PORT=... ./run_search.sh
set -e
cd "$(dirname "$0")"

PORT="${PORT:-8766}"
HOST="${HOST:-127.0.0.1}"

echo ""
echo "  Personal Database Search API → http://${HOST}:${PORT}/search"
echo "  Docs                         → http://${HOST}:${PORT}/docs"
echo "  Ctrl+C to stop."
echo ""

exec uv run uvicorn personal_db.search_app:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level warning
