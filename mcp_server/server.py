"""Enterprise MCP server with dummy Commercial Bank data.

Exposes four tools over MCP streamable HTTP (port 8001, path /mcp):
  employee_lookup, service_catalog, incident_records  - read-only
  update_incident_status                              - state-changing (the assistant
                                                        only calls it after human approval)

All data is fictional. State lives in memory, so restarts reset it.
Run:  python -m mcp_server.server
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "commercial-bank-enterprise-data",
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8001")),
    stateless_http=True,
)

EMPLOYEES = [
    {"name": "Nimal Perera", "title": "Head of Payments Engineering", "team": "Payments Platform",
     "email": "nimal.perera@combank.example", "on_call": False},
    {"name": "Dilini Fernando", "title": "SRE Lead", "team": "Site Reliability Engineering",
     "email": "dilini.fernando@combank.example", "on_call": True},
    {"name": "Kasun Jayasuriya", "title": "Mobile Payments Tech Lead", "team": "Digital Channels",
     "email": "kasun.jayasuriya@combank.example", "on_call": True},
    {"name": "Tharindu Silva", "title": "Card Systems Engineer", "team": "Cards",
     "email": "tharindu.silva@combank.example", "on_call": False},
    {"name": "Ishara Wickramasinghe", "title": "Information Security Manager", "team": "Security",
     "email": "ishara.w@combank.example", "on_call": False},
    {"name": "Ruwan Bandara", "title": "Integration Architect", "team": "Enterprise Architecture",
     "email": "ruwan.bandara@combank.example", "on_call": False},
]

SERVICES = [
    {"service": "Payment Switch", "owner_team": "Payments Platform", "tier": 1, "status": "operational",
     "depends_on": ["Core Banking", "HSM cluster", "LankaPay gateway"]},
    {"service": "Fund Transfer Engine", "owner_team": "Payments Platform", "tier": 1, "status": "operational",
     "depends_on": ["Payment Switch", "Core Banking"]},
    {"service": "Card Authorisation Service", "owner_team": "Cards", "tier": 1, "status": "degraded",
     "depends_on": ["Payment Switch", "Card scheme gateway"]},
    {"service": "Bill Payments Service", "owner_team": "Payments Platform", "tier": 2, "status": "operational",
     "depends_on": ["Fund Transfer Engine", "Biller API gateway"]},
    {"service": "Mobile Payments API", "owner_team": "Digital Channels", "tier": 1, "status": "operational",
     "depends_on": ["Fund Transfer Engine", "Payments DB"]},
]

INCIDENTS = {
    "INC-2026-013": {"incident_id": "INC-2026-013", "service": "Mobile Payments API", "severity": "SEV2",
                     "status": "resolved", "owner": "Kasun Jayasuriya"},
    "INC-2026-014": {"incident_id": "INC-2026-014", "service": "Card Authorisation Service", "severity": "SEV1",
                     "status": "monitoring", "owner": "Tharindu Silva"},
    "INC-2026-015": {"incident_id": "INC-2026-015", "service": "Card Authorisation Service", "severity": "SEV3",
                     "status": "open", "owner": "Dilini Fernando"},
    "INC-2026-016": {"incident_id": "INC-2026-016", "service": "Bill Payments Service", "severity": "SEV2",
                     "status": "open", "owner": "Nimal Perera"},
}


def _matches(value: str | None, wanted: str | None) -> bool:
    return wanted is None or (value or "").lower() == wanted.lower()


@mcp.tool()
def employee_lookup(query: str) -> dict:
    """Find employees by name, title or team."""
    q = query.lower()
    hits = [e for e in EMPLOYEES if q in e["name"].lower() or q in e["title"].lower() or q in e["team"].lower()]
    return {"query": query, "employees": hits}


@mcp.tool()
def service_catalog(service: str | None = None) -> dict:
    """Service ownership, tier, dependencies and current status."""
    items = [s for s in SERVICES if service is None or service.lower() in s["service"].lower()]
    return {"services": items}


@mcp.tool()
def incident_records(service: str | None = None, severity: str | None = None,
                     status: str | None = None) -> dict:
    """Live incident records, optionally filtered."""
    items = [i for i in INCIDENTS.values()
             if (service is None or service.lower() in i["service"].lower())
             and _matches(i["severity"], severity) and _matches(i["status"], status)]
    return {"incidents": items}


@mcp.tool()
def update_incident_status(incident_id: str, status: str, note: str) -> dict:
    """Change an incident's status (administrative action)."""
    if incident_id not in INCIDENTS:
        raise ValueError(f"unknown incident {incident_id}")
    if status not in ("open", "monitoring", "resolved"):
        raise ValueError("status must be open, monitoring or resolved")
    record = INCIDENTS[incident_id]
    previous = record["status"]
    record.update(status=status, last_note=note, updated_at=datetime.now(timezone.utc).isoformat())
    return {"incident_id": incident_id, "previous_status": previous, "new_status": status}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
