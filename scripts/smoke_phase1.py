"""Phase 1 smoke test: ingest one doc, retrieve, answer with citation.

Run with: uv run python scripts/smoke_phase1.py
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from rich import print as rprint

from personal_db.config import config
from personal_db.stores import configure_llama_index, init_stores


def main() -> None:
    rprint("[bold cyan]Phase 1 smoke test[/bold cyan]")

    configure_llama_index()
    stores = init_stores()

    sample_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample.md"
    text = sample_path.read_text()
    doc = Document(
        text=text,
        metadata={"path": str(sample_path), "source": "fixture", "doc_id": "sample-1"},
    )
    rprint(f"  ingesting: {sample_path.name} ({len(text)} chars)")

    splitter = SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    storage_context = StorageContext.from_defaults(vector_store=stores.vector_store)
    index = VectorStoreIndex.from_documents(
        [doc],
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=False,
    )
    rprint("  [green]✓[/green] ingested")

    query = "What are the three pillars of Helix?"
    rprint(f"\n[bold]Q:[/bold] {query}")

    qe = index.as_query_engine(similarity_top_k=4)
    response = qe.query(query)

    rprint(f"\n[bold]A:[/bold] {response}")
    rprint("\n[bold]Citations:[/bold]")
    for i, node in enumerate(response.source_nodes, 1):
        path = node.metadata.get("path", "?")
        score = node.score if node.score is not None else 0.0
        snippet = node.text[:120].replace("\n", " ")
        rprint(f"  [{i}] {Path(path).name} (score={score:.3f})")
        rprint(f"      {snippet}...")


if __name__ == "__main__":
    main()
