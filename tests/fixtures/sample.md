# Sample Notes: Project Atlas

Atlas is a fictional open-source task tracker. This file is sample data for
exercising ingestion and the knowledge graph — nothing here describes a real
person, company, or product.

The system has three parts:

1. **CLI** — a command-line client written in Go.
2. **Sync service** — a background service backed by PostgreSQL.
3. **Dashboard** — a web UI built with React.

Dana started Atlas in 2023 as a side project to learn distributed systems.
Atlas uses Redis for caching, runs its test suite with pytest, and integrates
with GitHub. Users can export their data as CSV at any time, so they are never
locked in.

A design note recorded here: coherence checks warn when a new change
contradicts an earlier decision, but the user can always override.
