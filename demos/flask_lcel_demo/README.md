# Flask + LCEL Demo API

A minimal API that demonstrates three LangChain/LCEL patterns served through Flask, powered by a local Ollama model. No frontend — just `curl`.

## Prerequisites

| Tool | Install |
|------|---------|
| **Python 3.10+** | [python.org](https://www.python.org/downloads/) |
| **Ollama** | [ollama.com/download](https://ollama.com/download) |

## Quick Start

```bash
# 1. Pull a model (one-time)
ollama pull llama3.2

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py
```

The server starts at `http://localhost:5000`.

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Which Ollama model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |

Example with a different model:

```bash
OLLAMA_MODEL=mistral python app.py
```

---

## API Endpoints

### `POST /api/chat` — Basic Chat

A simple sequential LCEL chain: `prompt | llm | parser`.

```bash
curl -s -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quantum computing in 2 sentences"}' | python -m json.tool
```

**Response:**

```json
{
    "response": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, unlike classical bits which are either 0 or 1. This allows quantum computers to solve certain complex problems exponentially faster than traditional computers."
}
```

---

### `POST /api/summarize` — Summarize Text

Demonstrates a specialized prompt template that constrains the LLM to return bullet points.

```bash
curl -s -X POST http://localhost:5000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "LangChain is an open-source framework designed to simplify the creation of applications using large language models. It provides modular components like chains, agents, and memory that can be composed together. The framework supports multiple LLM providers and includes tools for document loading, text splitting, and vector storage."
  }' | python -m json.tool
```

**Response:**

```json
{
    "summary": "- LangChain is an open-source framework for building LLM applications\n- It offers modular components (chains, agents, memory) for composition\n- Supports multiple providers and includes document/vector tools"
}
```

---

### `POST /api/analyze` — Parallel Analysis

Demonstrates `RunnableParallel` — three LCEL chains run concurrently on the same input, each with a different prompt.

```bash
curl -s -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The new product launch exceeded all expectations. Sales doubled in the first week, and customer feedback has been overwhelmingly positive. The team worked incredibly hard to make this happen."
  }' | python -m json.tool
```

**Response:**

```json
{
    "summary": "The product launch was highly successful with doubled sales and positive feedback.",
    "sentiment": "positive",
    "keywords": "product launch, sales, customer feedback, team, success"
}
```

---

### `GET /health` — Health Check

```bash
curl -s http://localhost:5000/health | python -m json.tool
```

```json
{
    "status": "ok",
    "model": "llama3.2"
}
```

---

## Architecture

```
curl ──POST──▶ Flask (/api/chat)       ──▶ ChatPromptTemplate ──▶ ChatOllama ──▶ StrOutputParser ──▶ JSON response
curl ──POST──▶ Flask (/api/summarize)   ──▶ ChatPromptTemplate ──▶ ChatOllama ──▶ StrOutputParser ──▶ JSON response
curl ──POST──▶ Flask (/api/analyze)     ──▶ RunnableParallel ─┬──▶ summary chain  ──┐
                                                              ├──▶ sentiment chain ──┤──▶ JSON response
                                                              └──▶ keywords chain  ──┘
```

## What Each Endpoint Teaches

| Endpoint | LCEL Pattern | Key Concept |
|---|---|---|
| `/api/chat` | `prompt \| llm \| parser` | Basic sequential chain |
| `/api/summarize` | `prompt \| llm \| parser` | Prompt engineering (constrained output) |
| `/api/analyze` | `RunnableParallel(...)` | Parallel execution with shared input |
