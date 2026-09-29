"""Remove duplicate chunk vectors from Qdrant.

Identity is (document content-hash, chunk text hash). The document hash lives
in _node_content.metadata — either under `pdb_doc_id` or, for points ingested
before that rename, plain `doc_id`. The TOP-LEVEL payload doc_id is useless
here: LlamaIndex overwrites it with a fresh per-ingest UUID, which is exactly
why the dedup gate never fired and the corpus was embedded four times.

Keeps, per group: a point carrying pdb_doc_id if one exists, else the most
recently ingested, tie-broken by id so the choice is deterministic.
"""
import hashlib, json, os
from collections import Counter, defaultdict

from qdrant_client.models import PointIdsList

from personal_db.config import config
from personal_db.stores import init_stores

DRY = os.environ.get("DRY", "1") == "1"

s = init_stores(); c = s.qdrant_client; coll = config.qdrant_collection

pts, off = [], None
while True:
    batch, off = c.scroll(coll, limit=512, with_payload=True, with_vectors=False, offset=off)
    pts.extend(batch)
    if off is None:
        break

groups = defaultdict(list)
for p in pts:
    pl = p.payload or {}
    meta = (json.loads(pl["_node_content"]).get("metadata") or {})
    text = json.loads(pl["_node_content"]).get("text") or ""
    did = meta.get("pdb_doc_id") or meta.get("doc_id")
    groups[(did, hashlib.sha1(text.encode()).hexdigest()[:16])].append(p)


def rank(p):
    """Higher sorts first: prefer a namespaced point, then the newest."""
    pl = p.payload or {}
    meta = json.loads(pl["_node_content"]).get("metadata") or {}
    return (1 if meta.get("pdb_doc_id") else 0, str(pl.get("ingested_at") or ""), str(p.id))


keep, drop = [], []
for _, members in groups.items():
    members = sorted(members, key=rank, reverse=True)
    keep.append(members[0])
    drop.extend(members[1:])

print(f"points {len(pts)}   groups {len(groups)}   keep {len(keep)}   delete {len(drop)}")
by_doc = Counter()
for p in drop:
    by_doc[(p.payload or {}).get("name", "?")] += 1
print("\ndeletions per document (top 10):")
for n, k in by_doc.most_common(10):
    print(f"  {k:5d}  {n}")

# Safety: every group must retain exactly one point, and no text may vanish.
kept_keys = {(json.loads((p.payload or {})['_node_content']).get('metadata') or {}).get('pdb_doc_id')
             or (json.loads((p.payload or {})['_node_content']).get('metadata') or {}).get('doc_id')
             for p in keep}
print(f"\ndistinct documents retained: {len(kept_keys)}")
assert len(keep) == len(groups), "a group lost its survivor"
assert len(keep) + len(drop) == len(pts), "point accounting mismatch"
print("invariants OK: every group keeps exactly one point; totals reconcile.")

if DRY:
    print("\nDRY RUN — nothing deleted. Set DRY=0 to apply.")
    raise SystemExit

ids = [p.id for p in drop]
for i in range(0, len(ids), 500):
    c.delete(collection_name=coll, points_selector=PointIdsList(points=ids[i:i + 500]))
    print(f"  deleted {min(i+500, len(ids))}/{len(ids)}", flush=True)

print("\nafter:", c.count(coll).count, "points")
