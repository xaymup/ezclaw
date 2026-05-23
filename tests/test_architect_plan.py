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


def test_execute_returns_intent_with_plan(fake_db):
    response = json.dumps({
        "kind": "execute",
        "current_task_id": 2,
        "recommended_agent": "executor",
        "reasoning": "Apply the cleanup hook found in task 1.",
        "plan": "1. Open sse_handler.py\n2. Find handle_disconnect\n3. Add connection.cleanup() before return\n4. Run tests",
        "task_updates": [{"id": 1, "status": "done"}],
        "new_tasks": [],
        "complete": False,
        "reflection": {
            "goal": "Fix SSE leak",
            "observation": "Read confirmed the leak location.",
            "critical_thinking": "Move to task 2.",
        },
    })
    arch = _make_architect_with_response(fake_db, response)
    plan = Plan(
        title="fix leak",
        tasks=[Task(id=1, description="read"), Task(id=2, description="fix"), Task(id=3, description="test")],
    )
    intent = arch.execute(plan, task_context="Step 1 read complete.")
    assert intent["kind"] == "execute"
    assert intent["current_task_id"] == 2
    assert intent["recommended_agent"] == "executor"
    assert intent["task_updates"] == [{"id": 1, "status": "done"}]
    assert intent["complete"] is False
    # plan field must be present and round-trip intact
    assert "plan" in intent
    assert "handle_disconnect" in intent["plan"]


def test_execute_returns_intent_without_plan(fake_db):
    """When plan is None, execute() still works — single-step path."""
    response = json.dumps({
        "kind": "execute",
        "current_task_id": 0,
        "recommended_agent": "general",
        "reasoning": "Conversational reply.",
        "task_updates": [],
        "new_tasks": [],
        "complete": True,
        "reflection": {"goal": "Answer", "observation": "", "critical_thinking": "Done."},
    })
    arch = _make_architect_with_response(fake_db, response)
    intent = arch.execute(None, task_context="User asked: hi")
    assert intent["kind"] == "execute"
    assert intent["complete"] is True


def test_execute_fallback_on_malformed_json(fake_db):
    """Two failed parses → return a safe fallback intent (no plan changes)."""
    from multi_agent import Architect

    with patch("multi_agent.build_architect_client") as build_client:
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "not json"}}
        build_client.return_value = (mock_client, "fake-model")
        arch = Architect(fake_db)

    intent = arch.execute(None, task_context="anything")
    # Fallback intent should be safe defaults, not raise
    assert isinstance(intent, dict)
    assert intent.get("recommended_agent") in ("executor", "general", "researcher", "debugger")
    assert intent.get("task_updates", []) == []
    assert intent.get("new_tasks", []) == []
    assert "plan" in intent
