# Personal Notes: Helix Architecture

Helix is an AI-native consultancy platform I am building. The core idea is that
intelligence should be portable, compounding, and governed.

The system has three pillars:

1. **Portability** — perspectives import from any major assistant, and export
   anywhere. Users are never locked to one provider.
2. **Compounding** — every decision, outcome, and challenge sharpens the system.
   Perspectives accrue credibility scores from real-world results, not popularity.
3. **Governance** — coherence checks warn when new actions contradict prior decisions
   or accepted doctrine. The user can override, but never silently.

I started Helix in mid-2024 after burning out on consulting engagements where every
deliverable was lost the moment the project ended. Knowledge had to live with the
client, not in my head and not in a deck.

The MCP server exposes the full intelligence surface to AI agents: ask, perspectives,
crucibles, discussions, doctrine, operate. The product language uses "Discussions"
publicly even though some legacy MCP names still say "pod".
