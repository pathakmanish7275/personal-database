"""Knowledge graph data layer over embedded Kuzu.

This owns the graph schema and all reads/writes. Extraction (which entities and
relations a chunk contains) lives in `personal_db.extract`; this module just
persists and queries the result, so it has no torch dependency and is fully
unit-testable on its own.

Schema:

    (Document)  doc_id, name, path, source, ingested_at
    (Chunk)     chunk_id, doc_id, text
    (Entity)    name (normalized key), label (display), type, mention_count, first_seen
    (Chunk)-[:PART_OF]->(Document)
    (Chunk)-[:MENTIONS {cnt}]->(Entity)
    (Entity)-[:RELATES_TO {predicate, confidence, source_chunk}]->(Entity)

Thread-safety: a Kuzu Connection is not thread-safe, but a Database is. We open
a short-lived Connection inside each public call so the web request thread and
the ingestion worker thread never share one. Writes are serialized upstream by
the global run lock, so concurrent writers don't happen.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import kuzu

log = logging.getLogger(__name__)


_SCHEMA = [
    "CREATE NODE TABLE IF NOT EXISTS Document(doc_id STRING, name STRING, path STRING, source STRING, ingested_at STRING, PRIMARY KEY(doc_id))",
    "CREATE NODE TABLE IF NOT EXISTS Chunk(chunk_id STRING, doc_id STRING, text STRING, PRIMARY KEY(chunk_id))",
    "CREATE NODE TABLE IF NOT EXISTS Entity(name STRING, label STRING, type STRING, mention_count INT64, first_seen STRING, PRIMARY KEY(name))",
    "CREATE REL TABLE IF NOT EXISTS PART_OF(FROM Chunk TO Document)",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Chunk TO Entity, cnt INT64)",
    "CREATE REL TABLE IF NOT EXISTS RELATES_TO(FROM Entity TO Entity, predicate STRING, confidence DOUBLE, source_chunk STRING)",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_key(name: str) -> str:
    """Normalized dedup key for an entity: collapsed whitespace, lowercased."""
    return " ".join((name or "").split()).lower()


def _is_useful(name: str) -> bool:
    key = _norm_key(name)
    return len(key) >= 2 and any(c.isalnum() for c in key)


def _norm_pred(predicate: str) -> str:
    return " ".join((predicate or "").split()).lower()[:80] or "related to"


class Graph:
    def __init__(self, db: kuzu.Database) -> None:
        self.db = db
        self._init_schema()

    def _conn(self) -> kuzu.Connection:
        return kuzu.Connection(self.db)

    def _init_schema(self) -> None:
        conn = self._conn()
        for ddl in _SCHEMA:
            conn.execute(ddl)

    # ─── writes (used by ingestion, single-threaded under the run lock) ──────

    def populate_document(self, doc: dict, chunks: list[dict]) -> dict:
        """Persist one document's graph contribution.

        `doc` = {doc_id, name, path, source, ingested_at}
        `chunks` = [{chunk_id, text, entities: [(name, type), ...],
                     relations: [(head, predicate, tail), ...]}]
        Returns a small summary of what was written."""
        conn = self._conn()
        self._add_document(conn, doc)
        n_entities = 0
        n_relations = 0
        for ch in chunks:
            self._add_chunk(conn, ch["chunk_id"], doc["doc_id"], ch.get("text", ""))
            for name, etype in ch.get("entities", []):
                if not _is_useful(name):
                    continue
                key = self._upsert_entity(conn, name, etype)
                self._link_mention(conn, ch["chunk_id"], key)
                n_entities += 1
            for head, predicate, tail in ch.get("relations", []):
                if not (_is_useful(head) and _is_useful(tail)):
                    continue
                if self._add_relation(conn, head, tail, predicate, ch["chunk_id"]):
                    n_relations += 1
        return {"entities": n_entities, "relations": n_relations, "chunks": len(chunks)}

    def _add_document(self, conn: kuzu.Connection, doc: dict) -> None:
        conn.execute(
            """MERGE (d:Document {doc_id: $id})
               ON CREATE SET d.name = $name, d.path = $path, d.source = $source, d.ingested_at = $ts
               ON MATCH  SET d.name = $name, d.path = $path""",
            {
                "id": doc["doc_id"],
                "name": doc.get("name") or "",
                "path": doc.get("path") or "",
                "source": doc.get("source") or "",
                "ts": doc.get("ingested_at") or _now(),
            },
        )

    def _add_chunk(self, conn: kuzu.Connection, chunk_id: str, doc_id: str, text: str) -> None:
        conn.execute(
            "MERGE (c:Chunk {chunk_id: $cid}) ON CREATE SET c.doc_id = $did, c.text = $text",
            {"cid": chunk_id, "did": doc_id, "text": (text or "")[:2000]},
        )
        conn.execute(
            """MATCH (c:Chunk {chunk_id: $cid}), (d:Document {doc_id: $did})
               MERGE (c)-[:PART_OF]->(d)""",
            {"cid": chunk_id, "did": doc_id},
        )

    def _ensure_entity(self, conn: kuzu.Connection, name: str, etype: str = "concept") -> str:
        key = _norm_key(name)
        conn.execute(
            """MERGE (e:Entity {name: $k})
               ON CREATE SET e.label = $label, e.type = $type, e.mention_count = 0, e.first_seen = $ts""",
            {"k": key, "label": name.strip(), "type": etype or "concept", "ts": _now()},
        )
        return key

    def _upsert_entity(self, conn: kuzu.Connection, name: str, etype: str) -> str:
        key = _norm_key(name)
        conn.execute(
            """MERGE (e:Entity {name: $k})
               ON CREATE SET e.label = $label, e.type = $type, e.mention_count = 1, e.first_seen = $ts
               ON MATCH  SET e.mention_count = e.mention_count + 1""",
            {"k": key, "label": name.strip(), "type": etype or "concept", "ts": _now()},
        )
        return key

    def _link_mention(self, conn: kuzu.Connection, chunk_id: str, entity_key: str) -> None:
        conn.execute(
            """MATCH (c:Chunk {chunk_id: $cid}), (e:Entity {name: $k})
               MERGE (c)-[m:MENTIONS]->(e)
               ON CREATE SET m.cnt = 1
               ON MATCH  SET m.cnt = m.cnt + 1""",
            {"cid": chunk_id, "k": entity_key},
        )

    def _add_relation(self, conn: kuzu.Connection, head: str, tail: str, predicate: str, source_chunk: str) -> bool:
        hk = self._ensure_entity(conn, head)
        tk = self._ensure_entity(conn, tail)
        if hk == tk:
            return False
        conn.execute(
            """MATCH (a:Entity {name: $h}), (b:Entity {name: $t})
               MERGE (a)-[r:RELATES_TO {predicate: $p}]->(b)
               ON CREATE SET r.confidence = $c, r.source_chunk = $sc""",
            {"h": hk, "t": tk, "p": _norm_pred(predicate), "c": 1.0, "sc": source_chunk or ""},
        )
        return True

    # ─── reads (used by the web layer, own connection per call) ──────────────

    def _scalar(self, conn: kuzu.Connection, query: str) -> int:
        res = conn.execute(query)
        return int(res.get_next()[0]) if res.has_next() else 0

    def counts(self) -> dict:
        conn = self._conn()
        return {
            "documents": self._scalar(conn, "MATCH (d:Document) RETURN count(d)"),
            "chunks": self._scalar(conn, "MATCH (c:Chunk) RETURN count(c)"),
            "entities": self._scalar(conn, "MATCH (e:Entity) RETURN count(e)"),
            "relations": self._scalar(conn, "MATCH ()-[r:RELATES_TO]->() RETURN count(r)"),
            "mentions": self._scalar(conn, "MATCH ()-[m:MENTIONS]->() RETURN count(m)"),
        }

    def top_entities(self, limit: int = 30) -> list[dict]:
        conn = self._conn()
        res = conn.execute(
            """MATCH (e:Entity)
               RETURN e.label, e.type, e.mention_count
               ORDER BY e.mention_count DESC, e.label LIMIT $lim""",
            {"lim": limit},
        )
        out = []
        while res.has_next():
            label, etype, mc = res.get_next()
            out.append({"name": label, "type": etype, "mentions": int(mc or 0)})
        return out

    def search_entities(self, query: str, limit: int = 30) -> list[dict]:
        conn = self._conn()
        res = conn.execute(
            """MATCH (e:Entity)
               WHERE e.name CONTAINS $q
               RETURN e.label, e.type, e.mention_count
               ORDER BY e.mention_count DESC, e.label LIMIT $lim""",
            {"q": _norm_key(query), "lim": limit},
        )
        out = []
        while res.has_next():
            label, etype, mc = res.get_next()
            out.append({"name": label, "type": etype, "mentions": int(mc or 0)})
        return out

    def entity_detail(self, name: str) -> dict | None:
        conn = self._conn()
        key = _norm_key(name)
        info = conn.execute(
            "MATCH (e:Entity {name: $k}) RETURN e.label, e.type, e.mention_count", {"k": key}
        )
        if not info.has_next():
            return None
        label, etype, mc = info.get_next()

        out_rels = []
        r = conn.execute(
            """MATCH (e:Entity {name: $k})-[r:RELATES_TO]->(o:Entity)
               RETURN r.predicate, o.label, o.type LIMIT 100""",
            {"k": key},
        )
        while r.has_next():
            pred, olabel, otype = r.get_next()
            out_rels.append({"predicate": pred, "target": olabel, "type": otype})

        in_rels = []
        r = conn.execute(
            """MATCH (e:Entity {name: $k})<-[r:RELATES_TO]-(o:Entity)
               RETURN o.label, o.type, r.predicate LIMIT 100""",
            {"k": key},
        )
        while r.has_next():
            olabel, otype, pred = r.get_next()
            in_rels.append({"predicate": pred, "source": olabel, "type": otype})

        docs = []
        r = conn.execute(
            """MATCH (e:Entity {name: $k})<-[:MENTIONS]-(c:Chunk)-[:PART_OF]->(d:Document)
               RETURN DISTINCT d.name, d.doc_id LIMIT 50""",
            {"k": key},
        )
        while r.has_next():
            dname, did = r.get_next()
            docs.append({"name": dname, "doc_id": did})

        return {
            "name": label,
            "type": etype,
            "mentions": int(mc or 0),
            "out_relations": out_rels,
            "in_relations": in_rels,
            "documents": docs,
        }

    # ─── query-time helpers (used by hybrid retrieval) ────────────────────

    def resolve_query_entities(self, names: list[str], limit_partial: int = 3) -> list[dict]:
        """Map raw query-entity strings to graph entities. Tries exact match on
        the normalized key first, then falls back to CONTAINS ranked by mentions."""
        conn = self._conn()
        out: list[dict] = []
        seen: set[str] = set()
        for raw in names:
            key = _norm_key(raw)
            if len(key) < 2:
                continue
            r = conn.execute(
                "MATCH (e:Entity {name: $k}) RETURN e.label, e.name, e.type, e.mention_count",
                {"k": key},
            )
            matched_exact = False
            while r.has_next():
                label, k2, etype, mc = r.get_next()
                if k2 in seen:
                    continue
                seen.add(k2)
                out.append({"label": label, "key": k2, "type": etype, "mentions": int(mc or 0)})
                matched_exact = True
            if matched_exact:
                continue
            r = conn.execute(
                """MATCH (e:Entity) WHERE e.name CONTAINS $k
                   RETURN e.label, e.name, e.type, e.mention_count
                   ORDER BY e.mention_count DESC LIMIT $lim""",
                {"k": key, "lim": limit_partial},
            )
            while r.has_next():
                label, k2, etype, mc = r.get_next()
                if k2 in seen:
                    continue
                seen.add(k2)
                out.append({"label": label, "key": k2, "type": etype, "mentions": int(mc or 0)})
        return out

    def chunks_for_entities(self, keys: list[str], limit: int = 10) -> list[dict]:
        """Chunks that mention any of these entity keys, ranked by aggregate
        mention weight across the set."""
        conn = self._conn()
        if not keys:
            return []
        where = " OR ".join(f"e.name = $k{i}" for i in range(len(keys)))
        params: dict = {f"k{i}": k for i, k in enumerate(keys)}
        params["lim"] = limit
        res = conn.execute(
            f"""MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity)
                WHERE {where}
                WITH c, sum(m.cnt) AS hits
                MATCH (c)-[:PART_OF]->(d:Document)
                RETURN c.chunk_id, c.text, c.doc_id, d.name, d.path, hits
                ORDER BY hits DESC LIMIT $lim""",
            params,
        )
        out = []
        while res.has_next():
            cid, text, did, dname, dpath, hits = res.get_next()
            out.append(
                {
                    "chunk_id": cid,
                    "text": text or "",
                    "doc_id": did,
                    "name": dname or "",
                    "path": dpath or "",
                    "hits": int(hits or 0),
                }
            )
        return out

    def relations_for_entities(self, keys: list[str], limit: int = 25) -> list[tuple[str, str, str]]:
        """Triples involving any of the given entity keys, in either direction."""
        conn = self._conn()
        if not keys:
            return []
        where = " OR ".join(f"a.name = $k{i} OR b.name = $k{i}" for i in range(len(keys)))
        params: dict = {f"k{i}": k for i, k in enumerate(keys)}
        params["lim"] = limit
        res = conn.execute(
            f"""MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                WHERE {where}
                RETURN a.label, r.predicate, b.label LIMIT $lim""",
            params,
        )
        out = []
        while res.has_next():
            out.append(tuple(res.get_next()))
        return out

    def graph_snapshot(self, limit: int = 120, max_edges: int = 400) -> dict:
        """Nodes + edges for the graph view.

        Takes the most-mentioned entities and keeps only relations whose *both*
        ends are in that set — a node-link view with dangling half-edges reads
        as broken, and edges to off-screen nodes carry no information. Degree is
        computed here so the renderer can size nodes by connectedness without a
        second pass over the data."""
        conn = self._conn()
        ents = self.top_entities(limit)
        by_key = {_norm_key(e["name"]): e for e in ents}

        res = conn.execute(
            """MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
               RETURN a.name, a.label, r.predicate, b.name, b.label
               LIMIT $lim""",
            {"lim": max_edges * 5},
        )
        edges: list[dict] = []
        degree: dict[str, int] = {}
        seen: set[tuple] = set()
        while res.has_next() and len(edges) < max_edges:
            ak, al, pred, bk, bl = res.get_next()
            if ak not in by_key or bk not in by_key or ak == bk:
                continue
            sig = (ak, bk, pred)
            if sig in seen:
                continue
            seen.add(sig)
            edges.append({"source": ak, "target": bk, "predicate": _norm_pred(pred or "")})
            degree[ak] = degree.get(ak, 0) + 1
            degree[bk] = degree.get(bk, 0) + 1

        nodes = [
            {
                "id": k,
                "label": e["name"],
                "type": e["type"] or "concept",
                "mentions": e["mentions"],
                "degree": degree.get(k, 0),
            }
            for k, e in by_key.items()
        ]
        return {"nodes": nodes, "edges": edges}

    def document_ids(self) -> set[str]:
        """doc_ids already present in the graph."""
        conn = self._conn()
        res = conn.execute("MATCH (d:Document) RETURN d.doc_id")
        out = set()
        while res.has_next():
            out.add(res.get_next()[0])
        return out

    def document_paths(self) -> set[str]:
        """File paths already represented in the graph (the stable backfill key —
        survives schema changes that affect how doc_id is computed)."""
        conn = self._conn()
        res = conn.execute("MATCH (d:Document) RETURN d.path")
        out = set()
        while res.has_next():
            p = res.get_next()[0]
            if p:
                out.add(p)
        return out
