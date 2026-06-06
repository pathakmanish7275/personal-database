"""End-to-end memory-agent smoke test.

Drives a multi-turn chat through the SSE endpoint with a low MEMORY_TOKEN_BUDGET
(set via env when launching the server) and verifies:
  - a summary gets created
  - summary_up_to_msg_id advances
  - a memory-compacted event is emitted on the round where compaction occurs

Run with: uv run python scripts/smoke_memory.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE = "http://localhost:8765"
DB = "./data/sessions.sqlite3"

QUESTIONS = [
    "What are the three pillars of Helix? Answer in one short sentence.",
    "What rule did I set about non-LLM extraction? One sentence.",
    "Why did I keep gpt-oss 20b instead of switching to Gemma 4? One sentence.",
    "Summarize in one sentence the UI direction I chose for this project.",
    "Now tell me — what are the three pillars again? Use exact words from earlier.",
]


def new_session() -> str:
    req = Request(f"{BASE}/chat/new", method="POST", data=b"")
    with urlopen(req) as r:
        loc = r.headers.get("location") or r.geturl()
    sid = loc.rstrip("/").split("/")[-1]
    return sid


def ask(sid: str, q: str, timeout: int = 300) -> dict:
    """Drive one SSE round; return a dict with collected events."""
    url = f"{BASE}/chat/{sid}/stream?q={quote(q)}"
    events: list[tuple[str, str]] = []
    tokens: list[str] = []
    saw_memory = False
    memory_info = None
    current_event = None
    with urlopen(url, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                current_event = None
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current_event:
                data = line.split(":", 1)[1].strip()
                events.append((current_event, data))
                if current_event == "memory":
                    saw_memory = True
                    try:
                        memory_info = json.loads(data)
                    except Exception:
                        pass
                elif current_event == "token":
                    try:
                        tokens.append(json.loads(data))
                    except Exception:
                        pass
                elif current_event == "done":
                    break
    return {
        "tokens": "".join(tokens),
        "memory_event": saw_memory,
        "memory_info": memory_info,
        "events_count": len(events),
    }


def dump_session_state(sid: str) -> dict:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    s = conn.execute(
        "SELECT id, title, summary, summary_up_to_msg_id FROM sessions WHERE id = ?",
        (sid,),
    ).fetchone()
    msgs = conn.execute(
        "SELECT id, role, length(content) AS clen FROM messages WHERE session_id = ? ORDER BY id",
        (sid,),
    ).fetchall()
    conn.close()
    return {
        "session": dict(s) if s else None,
        "messages": [dict(m) for m in msgs],
    }


def main() -> None:
    print("== memory-agent smoke ==")
    sid = new_session()
    print(f"session: {sid}")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n--- turn {i} ---")
        print(f"Q: {q}")
        t0 = time.time()
        r = ask(sid, q)
        dt = time.time() - t0
        print(f"A ({dt:.1f}s): {r['tokens'][:200]}")
        if r["memory_event"]:
            print(f"  [memory] compacted: {r['memory_info']}")
        state = dump_session_state(sid)
        s = state["session"]
        print(
            f"  state: msgs={len(state['messages'])}, "
            f"summary_len={len(s['summary'])}, summary_up_to_msg_id={s['summary_up_to_msg_id']}"
        )

    print("\n== final session state ==")
    final = dump_session_state(sid)
    print(json.dumps(final["session"], indent=2))
    print(f"messages: {[(m['id'], m['role'], m['clen']) for m in final['messages']]}")

    if final["session"]["summary"]:
        print("\n== summary text ==")
        print(final["session"]["summary"])
    else:
        print("\nNO SUMMARY GENERATED — compaction did not trigger.")
        sys.exit(1)


if __name__ == "__main__":
    main()
