# Design decisions, assumptions and trade-offs

## Model selection

| Role | Model | Why |
|---|---|---|
| Supervisor, research planner, aggregator, tools agent, response | **Claude Sonnet 5.5** (`claude-sonnet-5-5`) | Strong reasoning, reliable structured output and tool use, 1M context, at a mid-tier price. These steps decide routing, write code and produce the user-facing answer, so quality matters most here. |
| RLM batch analysts, conversation summaries | **Claude Haiku 5.5** (`claude-haiku-5-5`) | High volume and parallel; extraction from a small batch is an easy task. Roughly 20x cheaper than Sonnet, and lower latency per batch. |

Notes for this model generation, reflected in `app/agents/llm.py`: sampling parameters are not
sent (non-default values are rejected), and structured output uses native JSON-schema output
because forced tool choice is not supported. Swapping providers means changing one class.

Embeddings and reranking use Pinecone Inference (`llama-text-embed-v2`, `bge-reranker-v2-m3`),
so retrieval needs only the vector-DB key and stays independent of the LLM provider.

## Agent architecture

- **Supervisor with an explicit plan**, not a free-running ReAct loop. The supervisor returns an
  ordered list of steps; a router walks it. The behaviour is predictable, cheap, and visible in
  the state, trace and UI. The trade-off is less flexibility for open-ended multi-hop questions;
  the research agent covers those.
- **Tools agent split into planner and executor** so a human-approval `interrupt()` never re-runs
  the LLM on resume (LangGraph re-executes the interrupted node from the start).
- **Validator loop**: one regeneration with the validation issues as feedback, then deterministic
  repair (strip invalid citations, or a safe brand reply). Bounded cost, never loops.
- **State management**: turn-scoped list fields use a reducer where `None` resets, so the
  checkpointer can persist the conversation while each turn starts clean, and parallel writers
  (concurrent retrievals or batches) append safely.

## RLM implementation

The brief asks for the agent to explore, plan in Python, decompose, retrieve targeted sections,
recurse and aggregate, rather than load whole documents. Choices:

- The plan is **real Python** written by the model against a 4-function DSL, executed in an
  AST-validated sandbox. A deterministic default plan in the same DSL is used when the LLM is
  down or its plan is rejected, so the agent always has a plan.
- Batches are **regrouped by document** after the plan runs: an analyst sees whole incidents, and
  the number of LLM calls is bounded (`rlm_max_docs / rlm_batch_size`, default 24/4 = 6).
- **Fan-out with `Send`**: batches run in parallel; one failed batch doesn't fail the rest.
- **Theme counts are computed deterministically** from batch findings and given to the
  aggregator, so counts in the answer don't depend on the model's arithmetic.
- **Recursion** re-invokes the same compiled subgraph on at most one follow-up question, with a
  smaller document budget, bounded by `rlm_max_depth` (default 2).
- Simplification (allowed by the brief): plan code runs in-process. Production would use a
  container or microVM sandbox.

## Memory design

| Layer | Store | Scope | Purpose |
|---|---|---|---|
| Short-term | LangGraph `AsyncSqliteSaver` | session (thread) | full graph state, survives turns and restarts, enables resumable approvals |
| Rolling summary | graph state | session | beyond 12 messages, older turns are summarised and removed, bounding the context |
| Long-term | SQLite table | user, across sessions | profile, recent questions, related past Q&A recalled by keyword overlap |

Keyword recall is deliberate for the POC: it is deterministic, explainable in the activity panel,
and needs no extra embedding call. Only validated answers are remembered; blocked requests are
stored as a placeholder so injected text can't influence later turns.

## Security approach

Defence in depth; no single layer is trusted on its own:

1. **Authentication**: hardcoded users (option A), PBKDF2 hashes, constant-time compare, HS256
   JWT. The role is re-read from the server-side record on every request.
2. **Authorisation in code, never in the prompt**: document access levels become a Pinecone
   metadata filter (unauthorised chunks never reach the LLM); the model is shown only its role's
   tools; the executor re-checks every call (default deny). The research DSL and analysis tool
   only see role-filtered data. Conversation threads are namespaced by username.
3. **Prompt injection**:
   - *Instruction override / prompt extraction*: input patterns block the request.
   - *Data exfiltration*: patterns for sending data to URLs/emails and bulk dumps; no tool can
     send data outside the bank; secrets are redacted from answers.
   - *Tool abuse*: pydantic schemas with enums and patterns, free-text scanning, sandboxed code,
     timeouts, approval for state changes.
   - *Indirect injection*: retrieved text is sanitised (instruction-like lines removed) and
     wrapped in `<document>` tags that every prompt declares untrusted. The corpus contains a
     planted injection (MTG-003) to demonstrate this.
4. **Output validation**: citations must exist in the evidence (hallucinated citations are
   detected); brand rules for Commercial Bank (no investment advice, no commitments, no
   disparaging language); secret redaction.
5. **Rate limiting**: per-user token bucket (burst 10, 6/min sustained, configurable), with an
   exact `Retry-After`.

Pattern-based input checks are explainable and fast but can be evaded by paraphrase. That is why
they are only the first layer; authorisation and tool gates don't depend on detecting an attack.
A production system would add a classifier model as another input layer.

## Assumptions

- "Last year" in the RLM example is interpreted relative to today (the corpus spans Dec 2025 to Oct 2026).
- Hardcoded users are acceptable (brief option A). Keycloak would replace `authenticate()` only.
- One process, one region: in-memory rate limiter and circuit breakers. Multi-instance would
  move both to Redis.
- Synthetic corpus of 38 documents is enough to demonstrate the patterns; the pipeline is
  unchanged at thousands of documents (namespaces, metadata filters, catalog-first exploration).

## Trade-offs

| Decision | Benefit | Cost |
|---|---|---|
| Plan-driven supervisor | predictable, cheap, observable | less adaptive than a ReAct loop |
| In-process sandbox | simple, fast | weaker isolation than a container |
| Keyword long-term recall | deterministic, no extra calls | misses paraphrased history |
| SQLite for memory/checkpoints | zero setup, persistent | single-node only (Postgres saver is a drop-in) |
| Pinecone Inference embeddings | one vendor key, low latency | tied to Pinecone's model catalog |
| Regex guardrails | explainable, microseconds | evadable alone (mitigated by layers) |
