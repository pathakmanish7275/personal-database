"""Phase 1: extract the whole corpus to JSONL via OpenRouter.

Resumable: every chunk already present in the output file is skipped, so an
interrupted run continues where it stopped. Writes nothing to the graph — the
graph is built from this file in phase 2, which means filters can be retuned
later without paying for extraction again.
"""
import hashlib, json, os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from personal_db.config import config
from extractor import _SYSTEM, _parse, MAX_CHARS, MAX_TOKENS
from personal_db.llm_stream import stream_chat
from personal_db.stores import init_stores
from personal_db.ingest import list_documents

KEY     = os.environ["OPEN_ROUTER_KEY"]
# The ":free" slug was retired mid-project; OpenRouter 404s with "use this
# slug instead". This is the paid endpoint: ~$0.04/M in, $0.08/M out, which
# is roughly $0.28 for a full 3283-chunk corpus.
MODEL   = os.environ.get("OR_MODEL", "deepseek/deepseek-v4-flash-0731")
HOST    = "https://openrouter.ai/api"
OUT     = Path(os.environ.get("OUT", "data/kg-rebuild/extractions.jsonl"))
BATCH   = int(os.environ.get("BATCH", "50"))
WORKERS = int(os.environ.get("WORKERS", "8"))

_w = threading.Lock()
stats = Counter()


def chunk_id_for(path: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{path}|{idx}|{text[:200]}".encode()).hexdigest()[:16]
    return f"c_{h}"


def extract(text: str):
    """-> (entities, relations, ok). Retries 429/5xx with backoff."""
    for attempt in range(5):
        parts = []
        try:
            for ev in stream_chat(
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content": text[: MAX_CHARS]}],
                base_url=HOST, model=MODEL, api_key=KEY,
                max_tokens=MAX_TOKENS, timeout=180,
                extra_body={"reasoning": {"enabled": False}, "temperature": 0},
            ):
                if ev.kind == "content":
                    parts.append(ev.text)
            e, r = _parse("".join(parts))
            return e, r, True
        except Exception as exc:                                   # noqa: BLE001
            msg = str(exc)
            if attempt < 4 and any(c in msg for c in ("429", "500", "502", "503", "timeout", "Timeout")):
                time.sleep(min(2 ** attempt * 3, 30))
                continue
            with _w:
                stats[f"error:{type(exc).__name__}"] += 1
            return [], [], False
    return [], [], False


# ── build the work list ──────────────────────────────────────────────────────
stores = init_stores()
paths = sorted({d["path"] for d in list_documents(stores)
                if d.get("path") and Path(d["path"]).exists()})
splitter = SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

work = []
for p in paths:
    try:
        txt = Path(p).read_text(errors="ignore")
    except Exception:                                              # noqa: BLE001
        continue
    nodes = splitter.get_nodes_from_documents([Document(text=txt)])
    for i, n in enumerate(nodes):
        c = n.get_content()
        work.append({"path": p, "name": Path(p).name, "idx": i,
                     "chunk_id": chunk_id_for(p, i, c), "text": c})

done = set()
if OUT.exists():
    for line in OUT.open():
        try:
            row = json.loads(line)
        except Exception:                                          # noqa: BLE001
            continue
        # Only a successful extraction counts as done. Recording failures here
        # meant a retry skipped exactly the chunks that still needed work.
        if row.get("ok"):
            done.add(row["chunk_id"])
todo = [w for w in work if w["chunk_id"] not in done]

print(f"{len(paths)} files -> {len(work)} chunks; already done {len(done)}; to do {len(todo)}")
print(f"model={MODEL}  batch={BATCH}  workers={WORKERS}  out={OUT}\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
t_start = time.time()
fh = OUT.open("a")
for b in range(0, len(todo), BATCH):
    batch = todo[b: b + BATCH]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        res = list(pool.map(lambda w: extract(w["text"]), batch))
    for w, (e, r, ok) in zip(batch, res):
        if ok:
            fh.write(json.dumps({"chunk_id": w["chunk_id"], "path": w["path"],
                                 "name": w["name"], "idx": w["idx"], "text": w["text"],
                                 "entities": e, "relations": r, "ok": ok}) + "\n")
        stats["chunks"] += 1
        stats["entities"] += len(e)
        stats["relations"] += len(r)
        if not ok:
            stats["failed"] += 1
    fh.flush()
    n_done = b + len(batch)
    rate = (time.time() - t_start) / n_done
    print(f"  batch {b//BATCH + 1:2d}/{(len(todo)+BATCH-1)//BATCH}  "
          f"{n_done:4d}/{len(todo)} chunks  {time.time()-t0:5.1f}s  "
          f"entities {stats['entities']:5d}  relations {stats['relations']:5d}  "
          f"failed {stats['failed']:3d}  eta {(len(todo)-n_done)*rate/60:4.1f} min", flush=True)
fh.close()

print(f"\ndone in {(time.time()-t_start)/60:.1f} min")
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}")
