import pytest

from app.errors import ToolPermissionError
from app.retrieval.hybrid import SearchFilters, build_metadata_filter
from app.security.rbac import allowed_access_levels, allowed_tools, require_tool
from app.services import get_retriever


def test_tool_permissions_match_brief():
    assert allowed_tools("viewer") == ["deep_research", "knowledge_search"]
    assert "mcp_incident_records" in allowed_tools("analyst")
    assert "python_analysis" in allowed_tools("analyst")
    assert "mcp_update_incident_status" not in allowed_tools("analyst")
    assert "mcp_update_incident_status" in allowed_tools("admin")


def test_unknown_tools_are_denied():
    with pytest.raises(ToolPermissionError):
        require_tool("admin", "format_disk")


@pytest.mark.parametrize("role", ["viewer", "analyst", "admin"])
async def test_retrieval_never_returns_documents_above_clearance(role):
    result = await get_retriever().search("incident root cause certificate connection pool", role, top_k=20)
    assert result.hits
    assert result.degraded  # offline: local BM25 fallback, and the result says so
    assert {h.access_level for h in result.hits} <= set(allowed_access_levels(role))


async def test_viewer_cannot_see_confidential_incident():
    query = "Mobile Payments API connection pool"
    viewer_docs = {h.doc_id for h in (await get_retriever().search(query, "viewer", top_k=20)).hits}
    analyst_docs = {h.doc_id for h in (await get_retriever().search(query, "analyst", top_k=20)).hits}
    assert "INC-2026-008" not in viewer_docs  # confidential
    assert "INC-2026-008" in analyst_docs


def test_metadata_filter_shape():
    f = build_metadata_filter(("public", "internal"),
                              SearchFilters(document_types=["incident"], date_from="2026-01-01"))
    assert f == {"$and": [{"access_level": {"$in": ["public", "internal"]}},
                          {"document_type": {"$in": ["incident"]}},
                          {"created_ts": {"$gte": 20260101}}]}
