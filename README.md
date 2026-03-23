# Purple Builder Agent -- Build What I Mean

**Team TRIBUIA** | AgentBeats Phase 2 Sprint 1 | March 2026

A purple agent for the **build_what_i_mean** benchmark on [AgentBeats](https://agentbeats.dev) that constructs block structures on a 9x9x5 grid from natural-language instructions, combining LLM-powered spatial reasoning with pragmatic speaker modeling and strategic information gathering.

---

## Abstract

We present a purple agent for the AgentBeats *Build What I Mean* benchmark that interprets natural-language building instructions on a 9x9x5 block grid under scoring constraints where asking clarification questions incurs a penalty (-5 points) while correct and incorrect builds yield +10 and -10 respectively.

Our approach centers on three ideas:

1. **LLM-direct spatial reasoning with example-anchored prompting.** Rather than decomposing LLM output into intermediate plan representations and executing them deterministically, we let GPT-4o reason end-to-end from instruction to final block coordinates. The system prompt encodes the full coordinate system, direction semantics ("in front of" = +z, "behind" = -z), corner/edge positions, and six worked examples covering the most error-prone spatial patterns: corners with stacking, directional placement, rows from origin, chain references, and edge fills. This avoids the failure modes we observed when introducing pipeline complexity -- each additional processing stage introduced new bugs that degraded accuracy more than the stage's corrections improved it.

2. **Pragmatic speaker modeling.** The benchmark pairs the builder with two architect speakers who differ in how they resolve ambiguity: one uses consistent/conservative interpretations (same color as existing blocks, literal counts), the other is unpredictable. Our agent tracks per-speaker accuracy history and infers reliability from feedback on ambiguous rounds. After observing 2+ ambiguous instructions per speaker, it classifies the speaker as reliable (>60% conservative-correct rate) or unreliable, and adapts its interpretation strategy accordingly -- preferring conservative readings for reliable speakers and pattern-matching recent outcomes for unreliable ones.

3. **Cost-aware strategic questioning.** The agent asks questions only when the expected value of information exceeds the -5 penalty. For unknown speakers, it asks once on their first ambiguous instruction to efficiently learn their type (one -5 investment that saves multiple -10 errors). Once a speaker's reliability is determined, the agent stops asking and applies learned patterns, reducing the question rate to near zero for subsequent rounds. For fully specified instructions, the agent never asks regardless of speaker familiarity.

The agent also employs per-clause ambiguity detection that checks whether color and count are specified between each placing verb and its block noun (not just anywhere in the clause), coordinate validation with nearest-grid snapping, block deduplication, and graceful fallback to start-structure blocks on LLM failure.

---

## Architecture

```
src/
  server.py                 # A2A server entry point + AgentCard
  executor.py               # A2A protocol executor (task lifecycle)
  agent/
    builder_agent.py        # Core agent: LLM reasoning + strategy
    instruction_parser.py   # Message parsing + ambiguity detection
    speaker_model.py        # Per-speaker reliability tracking
    grid.py                 # Coordinate system + block utilities
```

### Pipeline

```
Green Agent Message
       |
       v
  Instruction Parser ──── Detect message type (instruction/feedback/answer/new game)
       |
       v
  Ambiguity Detector ──── Per-clause verb-to-noun color/count check
       |
       v
  Speaker Model ────────── Should we ask? (based on speaker history)
       |                         |
    [ASK]                    [BUILD]
       |                         |
       v                         v
  Question Generator        LLM Spatial Reasoning
  (LLM-based)               (GPT-4o, temp=0.0)
       |                         |
       v                         v
  Green answers              JSON block output
       |                         |
       v                         v
  LLM builds with answer    Validate + snap coords
       |                         |
       +────────+────────────────+
                |
                v
         [BUILD];Color,x,y,z;...
```

## Key Design Decisions

### Why LLM-Direct (No Deterministic Pipeline)

We initially built a complex pipeline with BAML typed outputs, a deterministic spatial executor, 18 enrichment rules, 4 auto-fix passes, and chain reference patching. Each component was individually sound, but the system scored progressively worse with each addition (73.75% -> 71.25% -> 63% -> 53%). The root cause: each processing stage introduced new edge-case bugs that outweighed its corrections. The LLM-direct approach avoids this by letting the model reason holistically about the instruction, producing final coordinates in one pass.

### Why Speaker Modeling

The benchmark has a hidden structure: one speaker always resolves ambiguity conservatively (same color, literal count), the other is unpredictable. By learning this from feedback, the agent can stop asking questions after a few rounds and still resolve ambiguity correctly for the reliable speaker. This saves -5 points per avoided question while maintaining accuracy.

### Why Minimal Asking

Our data shows that asking has positive expected value only when: (a) the instruction is genuinely ambiguous, AND (b) we don't yet know the speaker's pattern. Once we learn the speaker, guessing correctly is free (+10) while asking always costs (-5). The optimal strategy is to invest in 1-2 questions early to learn, then never ask again.

## Scoring

| Action | Points |
|--------|--------|
| Correct structure | +10 |
| Incorrect structure | -10 |
| Each question asked | -5 |

## Setup

```bash
# Prerequisites: Python 3.12+, uv package manager, OpenAI API key

# Install
uv sync

# Run locally
export OPENAI_API_KEY="your-key"
uv run src/server.py --host 127.0.0.1 --port 9018

# Docker
docker build -t purple-builder .
docker run -p 9018:9018 -e OPENAI_API_KEY="your-key" purple-builder
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key for LLM provider |
| `AGENT_MODEL` | `openai/gpt-4o` | LiteLLM model identifier |
| `AGENT_TEMPERATURE` | `0.0` | LLM temperature (0 = deterministic) |

## Team

- **Team TRIBUIA**
- Daniel Santiago Sandoval Higuera
- Julian Anibal Henao ([@julianAnibal](https://github.com/julianAnibal))

## License

MIT
