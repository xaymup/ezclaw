import json
import pytest
from unittest.mock import MagicMock, patch

from plan import Plan, Task


@pytest.fixture
def fake_db():
    return MagicMock()


def _make_architect_with_response(fake_db, response_text):
    """Construct an Architect with build_architect_client stubbed to return a
    mock client that yields `response_text` as the chat content."""
    from multi_agent import Architect

    with patch("multi_agent.build_architect_client") as build_client:
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": response_text}}
        build_client.return_value = (mock_client, "fake-model")
        arch = Architect(fake_db)
    return arch


def test_plan_returns_populated_plan_when_kind_is_plan(fake_db):
    response = json.dumps({
        "kind": "plan",
        "title": "fix the SSE memory leak",
        "tasks": [
            {"id": 1, "description": "Read sse_handler.py"},
            {"id": 2, "description": "Add cleanup in disconnect path"},
            {"id": 3, "description": "Add regression test"},
        ],
    })
    arch = _make_architect_with_response(fake_db, response)
    plan = arch.plan("fix the SSE memory leak")
    assert plan is not None
    assert plan.title == "fix the SSE memory leak"
    assert len(plan.tasks) == 3
    assert plan.tasks[0].description == "Read sse_handler.py"
    assert all(t.status == "pending" for t in plan.tasks)


def test_plan_returns_none_when_kind_is_single(fake_db):
    response = json.dumps({
        "kind": "single",
        "reason": "Conversational reply, no task list needed.",
    })
    arch = _make_architect_with_response(fake_db, response)
    plan = arch.plan("hello there")
    assert plan is None


def test_plan_returns_none_on_malformed_json_twice(fake_db):
    from multi_agent import Architect

    with patch("multi_agent.build_architect_client") as build_client:
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "not json at all"}}
        build_client.return_value = (mock_client, "fake-model")
        arch = Architect(fake_db)

    plan = arch.plan("anything")
    assert plan is None
    assert mock_client.chat.call_count >= 2
