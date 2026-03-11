# Memory Refactor Prompt

Read these files first:

- `simulation/persona/memory/associative_memory.py`
  Current memory store. It saves `nodes.json` and `embeddings.json`, defines `THOUGHT`, `EVENT`, `ACTION`, and retrieval-facing memory objects.
- `simulation/persona/cognition/store.py`
  Current write path. It creates thoughts/events/actions/chats, computes importance, and writes embeddings.
- `simulation/persona/cognition/retrieve.py`
  Current read path. It ranks memory with recency + importance + embedding relevance.
- `simulation/persona/persona.py`
  Wires each agent to `AssociativeMemory`, `StoreComponent`, and `RetrieveComponent`.
- `simulation/scenarios/fishing/agents/persona_v3/persona.py`
  Main agent loop in the election scenario. This is where retrieved memories are consumed.
- `simulation/scenarios/fishing/agents/persona_v3/cognition/utils.py`
  Shows the memory format that prompts already expect: dated plain-text memories.
- `simulation/main_elect.py`
  Creates the embedding model and passes it into the scenario.
- `simulation/scenarios/fishing/run_election.py`
  Builds the agents used by the election simulation.

Task:

I want to replace embedding-based memory/RAG with a single private `MEMORY.md` file per agent. Each agent should create its own `MEMORY.md`, append new memories to it, and retrieve from that file without using embedding similarity.

Please make a short implementation plan only. Do not write code yet.

The plan should answer:

1. Which files need to change first.
2. What the new `MEMORY.md` format should be.
3. How writes should work for thoughts/events/actions/chats.
4. How retrieval should work without embeddings.
5. Whether `EmbeddingModel` can be removed entirely or only bypassed at first.
6. What the smallest safe migration path is for the fishing election scenario.

Constraints:

- Keep the refactor narrow.
- Preserve current agent behavior as much as possible.
- Prefer append-only markdown over structured JSON unless a small index is absolutely necessary.
- Optimize for a simple first version, not a perfect long-term memory system.
