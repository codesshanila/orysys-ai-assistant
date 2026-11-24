from app.tools.registry import ToolContext, execute_tool


def ctx(role: str) -> ToolContext:
    return ToolContext(role=role, username=role, evidence=[])


async def test_viewer_cannot_call_mcp_tools():
    res = await execute_tool("mcp_incident_records", {}, ctx("viewer"))
    assert res.status == "denied"


async def test_unknown_tool_is_denied():
    assert (await execute_tool("shell", {"cmd": "ls"}, ctx("admin"))).status == "denied"


async def test_invalid_arguments_are_rejected():
    res = await execute_tool("mcp_update_incident_status",
                             {"incident_id": "DROP TABLE", "status": "resolved", "note": "x"}, ctx("admin"))
    assert res.status == "rejected"


async def test_python_analysis_sees_only_role_catalog():
    code = "result = sorted({d['access_level'] for d in catalog})"
    admin = await execute_tool("python_analysis", {"code": code}, ctx("admin"))
    analyst = await execute_tool("python_analysis", {"code": code}, ctx("analyst"))
    assert "restricted" in admin.output["result"]
    assert analyst.status == "ok" and "restricted" not in analyst.output["result"]


async def test_python_analysis_rejects_unsafe_code():
    res = await execute_tool("python_analysis", {"code": "import os\nresult = 1"}, ctx("analyst"))
    assert res.status == "error"


async def test_mcp_outage_is_reported_not_raised():
    # Nothing listens on the test MCP URL: the failure becomes data, not an exception.
    res = await execute_tool("mcp_service_catalog", {}, ctx("analyst"))
    assert res.status == "error"
