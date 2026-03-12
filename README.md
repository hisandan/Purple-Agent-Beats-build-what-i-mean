# Purple Builder Agent - AgentBeats Phase 2

A high-performance purple agent for the **build_what_i_mean** benchmark on [AgentBeats](https://agentbeats.dev). This agent constructs block structures in a 9×9×9 grid from natural-language instructions, using LLM-powered spatial reasoning with pragmatic speaker modeling.

## Architecture

```
src/
├── server.py                 # A2A server entry point + AgentCard
├── executor.py               # A2A protocol executor (task lifecycle)
└── agent/
    ├── builder_agent.py      # Core agent logic (LLM reasoning + strategy)
    ├── instruction_parser.py # Green agent message parser
    ├── speaker_model.py      # Pragmatic speaker reliability tracking
    └── grid.py               # Grid coordinate system utilities
```

## Approach

### LLM-Powered Spatial Reasoning
Uses GPT-4o (configurable via `AGENT_MODEL`) with a carefully crafted system prompt that encodes the full coordinate system, block placement rules, and output format. The LLM receives structured prompts with the instruction, existing blocks, and speaker context to produce validated block coordinates.

### Pragmatic Speaker Modeling
The benchmark pairs the agent with two speakers — one reliable (consistent interpretations) and one unreliable (mixed patterns). The agent:
- Tracks per-speaker accuracy history across rounds
- Infers reliability from feedback on ambiguous instructions
- Adapts interpretation strategy: conservative for reliable speakers, pattern-matching for unreliable ones

### Strategic Questioning
Each question costs -5 points, so the agent asks only when the expected information value exceeds the cost:
- Never asks on fully-specified instructions
- Asks once per unknown speaker on their first ambiguous instruction (to learn their type)
- Stops asking once speaker reliability is determined
- Uses LLM to generate targeted questions when asking

### Robust Error Handling
- Validates all coordinates against grid constraints
- Snaps invalid coordinates to nearest valid positions
- Falls back to start blocks on LLM failures
- Handles all message types (instructions, feedback, answers, transitions)

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key (or another LLM provider supported by LiteLLM)

### Install Dependencies
```bash
uv sync
```

### Run Locally
```bash
export OPENAI_API_KEY="your-key-here"
uv run src/server.py --host 127.0.0.1 --port 9018
```

### Configuration
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key for LLM provider |
| `AGENT_MODEL` | `openai/gpt-4o` | LiteLLM model identifier |
| `AGENT_TEMPERATURE` | `0.0` | LLM temperature (0 = deterministic) |

### Docker
```bash
# Build
docker build -t purple-builder-agent .

# Run
docker run -p 9018:9018 -e OPENAI_API_KEY="your-key" purple-builder-agent
```

## Evaluation

### Quick Submit (Recommended)
1. Register the agent on [agentbeats.dev](https://agentbeats.dev)
2. Go to the [build_what_i_mean](https://agentbeats.dev/agentbeater/build-what-i-mean) green agent page
3. Click "Quick Submit"
4. Select this purple agent and add your `OPENAI_API_KEY`
5. Submit and wait for results

### Manual Submit
```bash
# Edit scenario.toml with your agent IDs
# Then run locally:
pip install tomli-w requests
python generate_compose.py --scenario scenario.toml
cp .env.example .env  # Add OPENAI_API_KEY
mkdir -p output
docker compose up --abort-on-container-exit
```

## Scoring
| Action | Points |
|--------|--------|
| Correct structure | +10 |
| Incorrect structure | -10 |
| Each question asked | -5 |

## License
MIT
