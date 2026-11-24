"""Tool registry: every tool the agents can call, with a typed argument schema.

A tool call goes through four gates, in this order, inside `execute_tool`:
  1. Exists?            unknown tool names are rejected (default deny)
  2. Authorised?        rbac.require_tool(role, name), using the role from the verified
                        JWT. The model never decides authorisation.
  3. Valid arguments?   pydantic model per tool; free-text fields also pass the
                        tool-abuse/exfiltration scan in guardrails.validate_tool_args_text
  4. Bounded?           timeout per call; MCP calls also go through a circuit breaker
The result is a `ToolResult` that records status and timing, so failures become data the
response agent can mention ("the incident service was unavailable") instead of crashes.

Approval (human in the loop) for state-changing tools is enforced by the graph
(`tools` node, using rbac.APPROVAL_REQUIRED) before `execute_tool` is reached.
"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import get_settings
from app.errors import AssistantError, ToolPermissionError
from app.logging_setup import get_logger
from app.resilience import with_timeout
from app.security.guardrails import validate_tool_args_text
from app.security.rbac import require_tool
from app.tools.analysis import run_analysis
from app.tools.mcp_client import call_mcp_tool

log = get_logger(__name__)


# --- argument schemas -------------------------------------------------------------------
class _SafeText(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def _scan_text(cls, value):
        return validate_tool_args_text(value) if isinstance(value, str) else value


class KnowledgeSearchArgs(_SafeText):
    query: str = Field(..., min_length=2, max_length=300, description="What to search for")
    document_type: Literal["policy", "architecture", "runbook", "incident",
                           "meeting_notes", "product_spec"] | None = None


class PythonAnalysisArgs(BaseModel):
    code: str = Field(..., max_length=3000, description=(
        "Python over `catalog` (list of document rows: doc_id, title, document_type, "
        "department, service, severity, created_date) and `evidence` (retrieved chunks). "
        "Assign the answer to `result`. Counter is available. No imports."))


class EmployeeLookupArgs(_SafeText):
    query: str = Field(..., min_length=2, max_length=100, description="Name, team or role")


class ServiceCatalogArgs(_SafeText):
    service: str | None = Field(None, max_length=100, description="Service name; empty lists all")


class IncidentRecordsArgs(_SafeText):
    service: str | None = Field(None, max_length=100)
    severity: Literal["SEV1", "SEV2", "SEV3"] | None = None
    status: Literal["open", "monitoring", "resolved"] | None = None


class UpdateIncidentArgs(_SafeText):
    incident_id: str = Field(..., pattern=r"^INC-\d{4}-\d{3}$")
    status: Literal["open", "monitoring", "resolved"]
    note: str = Field(..., min_length=3, max_length=300)


# --- registry ---------------------------------------------------------------------------
@dataclass
class ToolContext:
    role: str
    username: str
    evidence: list[dict]


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[BaseModel, ToolContext], Awaitable[dict]]

    def anthropic_schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.args_model.model_json_schema()}


@dataclass
class ToolResult:
    name: str
    args: dict
    status: Literal["ok", "error", "denied", "rejected"]
    output: dict | None = None
    error: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


async def _knowledge_search(args: KnowledgeSearchArgs, ctx: ToolContext) -> dict:
    from app.retrieval.hybrid import SearchFilters
    from app.services import get_retriever

    filters = SearchFilters(document_types=[args.document_type] if args.document_type else None)
    result = await get_retriever().search(args.query, ctx.role, filters)
    return {"degraded": result.degraded,
            "hits": [{"chunk_id": h.chunk_id, "title": h.title, "section": h.section,
                      "text": h.text[:400]} for h in result.hits[:5]]}


async def _python_analysis(args: PythonAnalysisArgs, ctx: ToolContext) -> dict:
    return await run_analysis(args.code, ctx.role, ctx.evidence)


def _mcp(tool: str) -> Callable[[BaseModel, ToolContext], Awaitable[dict]]:
    async def run(args: BaseModel, ctx: ToolContext) -> dict:  # noqa: ARG001
        return await call_mcp_tool(tool, args.model_dump(exclude_none=True))
    return run


TOOLS: dict[str, ToolSpec] = {spec.name: spec for spec in [
    ToolSpec("knowledge_search", "Search the bank's indexed documents (hybrid search).",
             KnowledgeSearchArgs, _knowledge_search),
    ToolSpec("python_analysis", "Run a short, sandboxed Python analysis over the document "
             "catalog and retrieved evidence (counts, grouping, trends).",
             PythonAnalysisArgs, _python_analysis),
    ToolSpec("mcp_employee_lookup", "Look up employees, teams, on-call owners and contacts.",
             EmployeeLookupArgs, _mcp("employee_lookup")),
    ToolSpec("mcp_service_catalog", "Get service owners, tier, dependencies and status.",
             ServiceCatalogArgs, _mcp("service_catalog")),
    ToolSpec("mcp_incident_records", "Query live incident records (status, owner, severity).",
             IncidentRecordsArgs, _mcp("incident_records")),
    ToolSpec("mcp_update_incident_status", "Change an incident's status. Administrative; "
             "requires human approval.", UpdateIncidentArgs, _mcp("update_incident_status")),
]}


def tool_schemas_for(allowed: list[str]) -> list[dict]:
    """Only the tools a role may call are ever shown to the model."""
    return [TOOLS[n].anthropic_schema() for n in allowed if n in TOOLS]


async def execute_tool(name: str, raw_args: dict, ctx: ToolContext) -> ToolResult:
    started = time.perf_counter()

    def done(**kw) -> ToolResult:
        res = ToolResult(name=name, args=raw_args, elapsed_ms=int((time.perf_counter() - started) * 1000), **kw)
        log.info("tool_executed", tool=name, status=res.status, elapsed_ms=res.elapsed_ms, error=res.error)
        return res

    spec = TOOLS.get(name)
    if spec is None:
        return done(status="denied", error=f"unknown tool '{name}'")
    try:
        require_tool(ctx.role, name)
    except ToolPermissionError:
        return done(status="denied", error=f"role '{ctx.role}' may not use {name}")
    try:
        args = spec.args_model.model_validate(raw_args)
    except ValidationError as exc:
        first = exc.errors()[0]
        return done(status="rejected", error=f"invalid argument {first['loc']}: {first['msg']}")
    try:
        output = await with_timeout(spec.run(args, ctx), get_settings().tool_timeout_s, name)
    except AssistantError as exc:
        return done(status="error", error=f"{type(exc).__name__}: {exc.user_message}")
    return done(status="ok", output=output)
