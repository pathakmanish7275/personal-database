"""Prompt and parser for LLM knowledge-graph extraction.

Standalone on purpose: the app's ingest path uses the original GLiNER+REBEL
extractor, and nothing here is imported by `personal_db`. This is the separate
deepseek pipeline — see README.md.

The 13-label ontology is shared with the app's extractor so output stays
interchangeable; that is the one thing imported from it.
"""

from __future__ import annotations

from personal_db.extract.entities import LABELS

# A chunk is ~512 tokens. The cap bounds a runaway and keeps per-chunk cost
# predictable across a multi-thousand-chunk rebuild.
MAX_TOKENS = 700
MAX_CHARS = 4000

_SYSTEM = """You extract a knowledge graph from a passage of someone's personal notes.

Output ONE FACT PER LINE and nothing else. No prose, no JSON, no code fence.

  E|<name>|<type>
  R|<head>|<predicate>|<tail>

You are building a map of what this person works on and how those things fit
together — the kind of thing they would want to be reminded of in a year. You
are NOT documenting an API surface.

Rules:
- <type> must be exactly one of: %s
- Emit an E line for every entity worth remembering. Never invent one.
- Use the plain surface form. Strip backticks, quotes, asterisks and trailing
  punctuation from names.
- <predicate> is a short lowercase verb phrase of 1-3 words: "works at",
  "uses", "depends on", "part of", "decided to use". Never a sentence.
- Emit an R line for every SIGNIFICANT relationship the passage states or
  directly implies between two named entities — who owns, builds, uses,
  decided, works on, depends on, replaced, or is part of what. Prefer a few
  load-bearing relations over many shallow ones.
- Ground every relation in this passage. Do not guess from general knowledge,
  and do not invent a relation merely because two names appear together.
- ASCII diagrams and pipelines are relations: A -> B means A feeds B, not that
  A replaces B.
- If the passage names nothing real (boilerplate, a table of contents), output
  nothing at all.

NEVER emit these — they are true but worthless in a knowledge graph:
- Function parameters, arguments, struct fields, config keys, CLI flags or
  environment variable names, either as entities or as relations.
  NOT: R|change_stage|accepts argument|user_mobile
  NOT: R|config.yaml|contains|api_key
- A file or directory merely containing, or being located in, another.
  NOT: R|notes.md|part of|Projects/
- Pronouns, dates, version numbers, line counts, sizes, durations and bare
  numbers.
- Generic words that name no particular thing: "the system", "a tool",
  "the function", "the script", "the endpoint", "the database", "the config".
- A relation whose head and tail are the same thing under two spellings.

Prefer the specific over the containing thing: if a module uses a library, the
relation is module -> library, not repository -> library.

Example output:
E|Priya|person
E|deployment pipeline|project
E|Jenkins|tool
R|Priya|owns|deployment pipeline
R|deployment pipeline|uses|Jenkins
""" % ", ".join(LABELS)


_STRIP = " \t`\"'*_#,.;:()[]{}"


def _clean(s: str) -> str:
    return " ".join((s or "").strip().strip(_STRIP).split())


def _parse(raw: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    allowed = set(LABELS)
    entities: dict[str, tuple[str, str]] = {}      # lower(name) -> (name, type)
    pending: list[tuple[str, str, str]] = []

    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        kind, _, rest = line.partition("|")
        kind = kind.strip().upper()
        parts = [_clean(x) for x in rest.split("|")]

        if kind == "E" and len(parts) >= 2 and parts[0]:
            etype = parts[1].lower()
            entities.setdefault(
                parts[0].lower(), (parts[0], etype if etype in allowed else "concept")
            )
        elif kind == "R" and len(parts) >= 3 and all(parts[:3]):
            head, pred, tail = parts[0], parts[1].lower(), parts[2]
            if head.lower() != tail.lower():
                pending.append((head, pred, tail))

    # Relations may name something the model forgot to declare. Registering it
    # as a concept keeps the relation rather than silently dropping a real fact.
    for head, _pred, tail in pending:
        for name in (head, tail):
            entities.setdefault(name.lower(), (name, "concept"))

    return list(entities.values()), pending
