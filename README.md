# CodeForge 

**Autonomous repo builder — any AI provider.** Point it at Anthropic, OpenAI, OpenRouter, Google Gemini, Moonshot AI, Qwen, Mistral, Groq, Together AI, Ollama (local), or any OpenAI-compatible endpoint. It runs a continuous PERCEIVE → PLAN → EXECUTE → MOLT cycle until your repo is fully built.

---

## Quick Start

```bash
git clone <this-repo>
cd codeforge
pip install -e .

codeforge build "a FastAPI todo API with SQLite"
# → First run triggers the interactive onboarding wizard
```

On first run, the onboarding wizard walks you through:
1. Choosing a provider
2. Entering your API key
3. Picking a model
4. Setting defaults

---

## Supported Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **Anthropic** | Claude Opus, Sonnet, Haiku | Native tool use |
| **OpenAI** | GPT-4o, o3, o1 | Function calling |
| **OpenRouter** | 200+ models | One key, any model |
| **Google Gemini** | Gemini 2.0, 1.5 Pro/Flash | Requires `pip install -e ".[gemini]"` |
| **Moonshot AI** | Kimi 128k | OpenAI-compatible |
| **Alibaba Qwen** | Qwen-Max, Coder | OpenAI-compatible |
| **Mistral** | Large, Codestral | Codestral is great for this |
| **Groq** | Llama 3.3, Mixtral | Ultra-fast inference |
| **Together AI** | Llama, DeepSeek-R1 | Open-source at scale |
| **Ollama** | Any local model | No API key needed |
| **Custom** | Any OpenAI-compat endpoint | BYO endpoint |

---

## Install

```bash
# Core (Anthropic only)
pip install -e .

# With OpenAI-compatible providers (OpenAI, OpenRouter, Moonshot, Qwen, etc.)
pip install -e ".[openai]"

# With Google Gemini
pip install -e ".[gemini]"

# Everything
pip install -e ".[all]"
```

---

## Usage

### Build a new repo

```bash
codeforge build "a REST API for a todo app using FastAPI and SQLite"
codeforge build "a TypeScript Discord bot with slash commands" --lang typescript
codeforge build "a Rust CLI markdown-to-HTML converter" --out ./md2html
codeforge build "a Go gRPC auth service" --model gpt-4o
codeforge build "a Next.js app with Python FastAPI backend" --max-cycles 50
```

### Per-run provider/model override

```bash
# Use a different provider just for this run
codeforge build "build a scraper" --provider openrouter --model deepseek/deepseek-r1

# Use Codestral for a code-heavy task
codeforge build "a parser for custom DSL" --provider mistral --model codestral-latest

# Run locally with Ollama
codeforge build "a Python web scraper" --provider ollama --model qwen2.5-coder:32b
```

### Resume / inspect / test

```bash
codeforge resume ./my-project         # resume an interrupted session
codeforge inspect ./my-project        # view memory, plan, file map
codeforge test ./my-project           # test + self-critique pass
```

### Config management

```bash
codeforge config     # interactive setup / reconfigure
codeforge providers  # list all providers and their status
```

---

## Architecture

```
codeforge build "..."
       │
       ▼  [first run: onboarding wizard]
       │
       ▼
┌──────────────────────────────────────────────────┐
│  CYCLE (PERCEIVE → PLAN → EXECUTE → MOLT)        │
│                                                  │
│  ForgeMemory ────► system prompt injection       │
│       │                                          │
│  Provider.complete(system, messages, tools)      │
│  (Anthropic / OpenAI / Gemini / Ollama / ...)    │
│       │                                          │
│  ToolDispatcher                                  │
│    write_file │ read_file │ run_command          │
│    search │ update_plan │ write_agent_note       │
│       │                                          │
│  MOLT: save ForgeMemory → .forge/state.json      │
└──────────────────────────────────────────────────┘
       │
       ▼  [forge_complete called]
     DONE ✨
```

**ForgeMemory** is the agent's exoskeleton. It persists the plan, file map, agent notes, and a rolling 20-turn message window after every cycle. Interrupted builds resume exactly where they left off via `codeforge resume`.

---

## Config file

`~/.config/codeforge/config.json` — stores all provider configs. You can also set API keys via environment variables:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export GOOGLE_API_KEY=...
export MOONSHOT_API_KEY=...
export DASHSCOPE_API_KEY=...    # Qwen
export MISTRAL_API_KEY=...
export GROQ_API_KEY=...
export TOGETHER_API_KEY=...
```

The onboarding wizard detects these automatically.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT
