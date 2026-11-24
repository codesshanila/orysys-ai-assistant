# Commercial Bank Knowledge Assistant

An enterprise AI assistant that answers employees' questions from internal documents
(policies, architecture, runbooks, incident reports, product specs, meeting notes),
built as a **LangGraph multi-agent system** with a **Recursive Language Model (RLM)**
research agent, **Pinecone hybrid retrieval**, **LangSmith tracing**, role-based access
control, layered prompt-injection defences and graceful degradation.

Built for the Orysys AI Lead technical assessment. All documents and enterprise data are synthetic.

| | |
|---|---|
| Architecture & diagrams | [docs/architecture.md](docs/architecture.md) |
| Design decisions, assumptions, trade-offs | [docs/decisions.md](docs/decisions.md) |
| Demo script | [docs/demo-script.md](docs/demo-script.md) |

## What it does

- **Chat** with streaming answers, multi-turn context and inline citations (`[INC-2026-008#03]`).
- **Agent activity panel** showing, live: the active graph node, the supervisor's plan, retrieval
  scores (dense / sparse / hybrid / rerank), the research agent's generated Python search plan and
  parallel batches, tool calls, memory updates and validation results.
- **Specialised agents**: supervisor → retrieval / research (RLM) / tools → response → validator.
- **RLM research**: for "summarise all payment outages and find recurring root causes", the agent
  explores the catalog, writes a sandboxed Python search plan, pulls only the relevant sections,
  analyses document batches in parallel, aggregates the themes and recursively investigates a
  follow-up question.
- **Tools**: knowledge search, sandboxed Python analysis, and an MCP server with an employee
  directory, service catalog and incident records (plus an approval-gated status update).
- **Security**: JWT auth, RBAC on documents *and* tools, input/retrieval/output guardrails,
  citation verification, brand rules, per-user token-bucket rate limiting.
- **Resilience**: every dependency (LLM, Pinecone, MCP, tools) has a timeout, a circuit breaker
  and a fallback; administrators can inject failures from the UI to demonstrate it.

## Quick start

Requirements: Python 3.13, plus API keys for Anthropic, Pinecone and LangSmith.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
cp .env.example .env              # Windows cmd: copy .env.example .env; then fill in the keys
python scripts/ingest.py          # chunk, build BM25 stats, embed and upsert to Pinecone
```

On Windows, clone into a short path (for example `C:\orysys`) or enable
[long path support](https://pip.pypa.io/warnings/enable-long-paths): the Anthropic SDK has very
long file names and `pip install` fails inside deeply nested folders.

`data/index/` (chunks, catalog, BM25 statistics) is committed, so offline mode and the tests need
no ingest. With your own Pinecone key, run `ingest.py` once to create and fill the index.

Run the three processes (separate terminals):

```bash
python -m mcp_server.server                       # MCP server  :8001
uvicorn app.api.main:app --port 8000              # backend     :8000  (OpenAPI at /docs)
streamlit run frontend/streamlit_app.py           # frontend    :8501
```

Or with Docker (tested with Docker Desktop 29 and Compose v5 on Windows/WSL 2):

```bash
docker compose up --build        # needs .env; frontend :8501, backend :8000, MCP internal only
```

Stop any local servers first so ports 8000 and 8501 are free. The backend waits for MCP, and the
frontend waits for a healthy backend. Conversation checkpoints and long-term memory live in the
`runtime` volume, so they survive restarts. Use `docker compose down -v` to clear them.

### Demo users (brief option A: hardcoded users and roles)

| User | Password | Role | Can |
|---|---|---|---|
| `viewer` | `viewer123` | Viewer | chat, search (public + internal documents) |
| `analyst` | `analyst123` | Analyst | + Python analysis, MCP read tools, confidential documents |
| `admin` | `admin123` | Administrator | all tools (incident updates with approval), restricted documents, failure injection |

Passwords are stored as salted PBKDF2 hashes in `app/security/auth.py`.

### Offline mode

With no keys the system still runs: the LLM steps use deterministic fallbacks (keyword routing,
extractive answers with citations, heuristic batch analysis) and retrieval uses the local BM25
index. The test suite runs this way:

```bash
python -m pytest -q        # 63 tests, ~10 s, no network
```

## Project layout

```
app/
  agents/      graph.py (wiring), nodes.py (agents), research.py (RLM subgraph), state.py,
               llm.py (Claude gateway), memory.py, prompts.py, events.py (activity stream)
  retrieval/   documents.py (chunking, catalog), bm25.py, pinecone_store.py, hybrid.py
  security/    auth.py, rbac.py, guardrails.py, sandbox.py, rate_limit.py
  tools/       registry.py (schemas + 4 gates), mcp_client.py, analysis.py
  api/main.py  FastAPI: auth, SSE chat, HITL resume, feedback, chaos, health
  config.py, errors.py, resilience.py, logging_setup.py
mcp_server/    FastMCP server with dummy enterprise data
frontend/      Streamlit UI
scripts/       generate_documents.py (synthetic corpus), ingest.py
data/          documents/ (38 markdown docs), index/ (chunks, catalog, BM25 stats)
tests/         offline unit + end-to-end tests
```

## Observability

Set `LANGSMITH_API_KEY` and every turn is traced in the LangSmith project `orysys-ai-assistant`:
one root run per turn (tagged with user role and session), with child runs for each graph node,
each LLM call, the research subgraph and every parallel batch, the `hybrid_search` retriever runs
(documents with dense/sparse/rerank scores) and tool calls. Thumbs up/down in the UI is attached to
the turn's run as `user_score` feedback. Logs are structured JSON carrying the same request id.
