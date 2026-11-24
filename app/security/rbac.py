"""Role-based access control: one policy table, enforced in code.

Two things are controlled per role:
  1. Which document classifications retrieval may return (applied as a Pinecone metadata
     filter, so unauthorised chunks never reach the LLM at all).
  2. Which tools may run. The LLM is only *shown* the tools its user may call, and the
     tool executor re-checks every call - so even a model tricked into requesting a
     forbidden tool cannot run it. The model never decides authorisation.
"""
from __future__ import annotations

from enum import Enum

from app.errors import ToolPermissionError


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


ACCESS_LEVELS_BY_ROLE: dict[Role, tuple[str, ...]] = {
    Role.VIEWER: ("public", "internal"),
    Role.ANALYST: ("public", "internal", "confidential"),
    Role.ADMIN: ("public", "internal", "confidential", "restricted"),
}

# Tool name -> roles allowed. Brief: viewer = chat + search; analyst = search, analytics,
# MCP tools; administrator = all tools.
TOOL_PERMISSIONS: dict[str, frozenset[Role]] = {
    "knowledge_search": frozenset({Role.VIEWER, Role.ANALYST, Role.ADMIN}),
    "deep_research": frozenset({Role.VIEWER, Role.ANALYST, Role.ADMIN}),
    "python_analysis": frozenset({Role.ANALYST, Role.ADMIN}),
    "mcp_employee_lookup": frozenset({Role.ANALYST, Role.ADMIN}),
    "mcp_service_catalog": frozenset({Role.ANALYST, Role.ADMIN}),
    "mcp_incident_records": frozenset({Role.ANALYST, Role.ADMIN}),
    # Administrative tools: change state, so they also require human approval (HITL).
    "mcp_update_incident_status": frozenset({Role.ADMIN}),
    "admin_chaos_control": frozenset({Role.ADMIN}),
}

# Tools that modify data. They pause the graph for explicit human approval.
APPROVAL_REQUIRED = frozenset({"mcp_update_incident_status"})


def allowed_access_levels(role: Role | str) -> tuple[str, ...]:
    return ACCESS_LEVELS_BY_ROLE[Role(role)]


def allowed_tools(role: Role | str) -> list[str]:
    r = Role(role)
    return sorted(name for name, roles in TOOL_PERMISSIONS.items() if r in roles)


def can_use_tool(role: Role | str, tool: str) -> bool:
    return Role(role) in TOOL_PERMISSIONS.get(tool, frozenset())


def require_tool(role: Role | str, tool: str) -> None:
    """Raise unless `role` may run `tool`. Unknown tools are denied (default deny)."""
    if not can_use_tool(role, tool):
        raise ToolPermissionError(f"role={role} denied tool={tool}")
