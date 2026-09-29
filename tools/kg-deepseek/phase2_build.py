"""Phase 2: build a graph from the phase-1 JSONL. No API calls.

Filtering happens here rather than in the prompt, because the model does not
reliably honour exclusion rules (measured: the tightened prompt cut relations
by only 7.5% and kept emitting the patterns it was told to omit). A rule in
code is deterministic and can be retuned by re-running this file.
"""
import json, os, re, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from pathlib import Path

SRC  = Path(os.environ.get("SRC", "data/kg-rebuild/extractions.jsonl"))
DRY  = os.environ.get("DRY", "1") == "1"
DEST = Path(os.environ.get("DEST", "data/kg-rebuild/kuzu/personal.db"))

FILE_EXT = re.compile(r"\.(py|js|ts|tsx|jsx|md|ya?ml|json|txt|toml|ini|cfg|sh|env|lock|cu|go|rs|sql|html|css)$", re.I)
CONSTY   = re.compile(r"^[A-Z][A-Z0-9_]{3,}$")
CONTAIN  = {"contains", "part of", "has", "includes", "lists", "omits", "lacks",
            "located in", "inside", "is part of"}


def pathy(s: str) -> bool:
    """A filesystem path, not a name that merely contains a slash.

    The space test matters: "GPT-4o Realtime/Azure" is a product name, and an
    earlier version of this rule threw it away."""
    if " " in s or s.lower().startswith("http"):
        return False
    return "/" in s or s.startswith(".")


def filey(s: str) -> bool:
    return bool(FILE_EXT.search(s)) or pathy(s)


REASONS = {
    "containment of a file/path": lambda h, p, t: p in CONTAIN and (filey(h) or filey(t)),
    "endpoint is a filesystem path": lambda h, p, t: pathy(h) or pathy(t),
    "endpoint is an ALL_CAPS constant": lambda h, p, t: bool(CONSTY.match(h) or CONSTY.match(t)),
    "self-relation": lambda h, p, t: h.strip().lower() == t.strip().lower(),
}

rows = [json.loads(l) for l in SRC.open()]
# Test fixtures must never reach the live graph: tests/fixtures/sample.md was
# found sitting in the vector store as a real document, and its content is
# invented, so retrieval could state it back as fact.
rows = [r for r in rows if "/tests/" not in r["path"]]
tot_r = sum(len(r["relations"]) for r in rows)
tot_e = sum(len(r["entities"]) for r in rows)

dropped = Counter()
kept_rows = []
for row in rows:
    keep = []
    for h, p, t in row["relations"]:
        why = next((k for k, f in REASONS.items() if f(h, p, t)), None)
        if why:
            dropped[why] += 1
        else:
            keep.append((h, p, t))
    # An entity earns its place by being in a surviving relation, or by being a
    # person/organization (those are worth keeping even when isolated).
    live = {x.lower() for h, _, t in keep for x in (h, t)}
    ents = [(n, ty) for n, ty in row["entities"]
            if n.lower() in live or ty in ("person", "organization")]
    kept_rows.append({**row, "relations": keep, "entities": ents})

kept_r = sum(len(r["relations"]) for r in kept_rows)
kept_e = sum(len(r["entities"]) for r in kept_rows)

print(f"source: {len(rows)} chunks, {tot_e} entities, {tot_r} relations")
print("\nrelations dropped:")
for k, v in dropped.most_common():
    print(f"  {v:5d}  {k}")
print(f"  {sum(dropped.values()):5d}  TOTAL ({sum(dropped.values())/tot_r*100:.1f}%)")
print(f"\nkept: {kept_e} entities ({kept_e/tot_e*100:.0f}%), "
      f"{kept_r} relations ({kept_r/tot_r*100:.0f}%)")
print(f"      {kept_r/len(rows):.1f} relations/chunk")

uniq_r = {tuple(x) for r in kept_rows for x in r["relations"]}
uniq_e = {n.lower() for r in kept_rows for n, _ in r["entities"]}
print(f"\nafter dedup across chunks: {len(uniq_e)} distinct entities, "
      f"{len(uniq_r)} distinct relations")
print(f"  (current graph: 2624 entities, 1221 relations)")

if DRY:
    print("\nDRY RUN — nothing written. Set DRY=0 to build the graph.")
    raise SystemExit

# ── write a fresh graph ──────────────────────────────────────────────────────
import kuzu
from personal_db.kg import Graph
from personal_db.ingest import _doc_id_for

if DEST.parent.exists():
    shutil.rmtree(DEST.parent)
DEST.parent.mkdir(parents=True, exist_ok=True)
graph = Graph(kuzu.Database(str(DEST)))

by_doc: dict[str, list] = {}
for row in kept_rows:
    by_doc.setdefault(row["path"], []).append(row)

written = Counter()
for path, rws in sorted(by_doc.items()):
    p = Path(path)
    text = p.read_text(errors="ignore")
    doc_id = _doc_id_for(p, text)
    doc = {"doc_id": doc_id, "name": p.name, "path": str(p.resolve()),
           "source": p.suffix.lstrip(".").lower(),
           "ingested_at": __import__("datetime").datetime.now(
               __import__("datetime").timezone.utc).isoformat()}
    chunks = [{"chunk_id": r["chunk_id"], "text": r["text"],
               "entities": [tuple(x) for x in r["entities"]],
               "relations": [tuple(x) for x in r["relations"]]} for r in rws]
    res = graph.populate_document(doc, chunks)
    written["entities"] += res["entities"]
    written["relations"] += res["relations"]
    print(f"  {p.name[:56]:56s} {len(chunks):4d} chunks  "
          f"{res['entities']:5d}e {res['relations']:5d}r")

print(f"\nwrote {DEST}")
print("counts:", graph.counts())
