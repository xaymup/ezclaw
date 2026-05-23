"""Tier 2.2 — nested plan rendering tests.

Sub-tasks (Task.parent_id != None) should render indented under their
parent rather than as flat siblings. The architect prompt now teaches
the model to emit `parent_id` in `new_tasks` when a follow-up is a
sub-step of an existing task."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plan import Plan, Task


def test_task_dataclass_has_parent_id_field():
    """The Task dataclass exposes parent_id with a default of None so
    existing plans (no parent set) keep working unchanged."""
    t = Task(id=1, description="x")
    assert hasattr(t, "parent_id")
    assert t.parent_id is None


def test_task_can_be_constructed_with_explicit_parent():
    t = Task(id=2, description="sub-step", parent_id=1)
    assert t.parent_id == 1


def test_plan_insert_passes_parent_id_through():
    """Plan.insert(after_id, desc, parent_id=X) attaches the new task
    to parent X. Without parent_id, the task is a top-level sibling."""
    plan = Plan(title="t", tasks=[Task(id=1, description="parent")])

    # Sub-task: parent_id set explicitly
    sub = plan.insert(1, "discovered sub-step", parent_id=1)
    assert sub.parent_id == 1

    # Sibling: parent_id omitted
    sib = plan.insert(1, "another top-level step")
    assert sib.parent_id is None


def test_plan_roots_returns_only_top_level_tasks():
    plan = Plan(title="t", tasks=[
        Task(id=1, description="root A"),
        Task(id=2, description="child of A", parent_id=1),
        Task(id=3, description="root B"),
        Task(id=4, description="child of B", parent_id=3),
        Task(id=5, description="grandchild of A (child of 2)", parent_id=2),
    ])
    roots = plan.roots()
    assert [t.id for t in roots] == [1, 3]


def test_plan_children_of_returns_direct_children_only():
    plan = Plan(title="t", tasks=[
        Task(id=1, description="root"),
        Task(id=2, description="child", parent_id=1),
        Task(id=3, description="another child", parent_id=1),
        Task(id=4, description="grandchild", parent_id=2),
    ])
    assert [t.id for t in plan.children_of(1)] == [2, 3]
    assert [t.id for t in plan.children_of(2)] == [4]
    assert plan.children_of(3) == []


def test_plan_walk_handles_arbitrary_depth():
    """The walker has to recurse — a 3-deep tree must surface
    grandchildren via children_of on the middle node."""
    plan = Plan(title="t", tasks=[
        Task(id=1, description="A"),
        Task(id=2, description="A.1", parent_id=1),
        Task(id=3, description="A.1.a", parent_id=2),
    ])
    # Simulate the renderer's walk
    visited: list = []

    def walk(task_id, depth=0):
        visited.append((task_id, depth))
        for child in plan.children_of(task_id):
            walk(child.id, depth + 1)

    for root in plan.roots():
        walk(root.id, 0)

    assert visited == [(1, 0), (2, 1), (3, 2)]


def test_existing_flat_plan_still_works():
    """No parent_id anywhere → tree walk degenerates to flat iteration,
    preserving backward compatibility with plans built before Tier 2.2."""
    plan = Plan(title="t", tasks=[
        Task(id=1, description="a"),
        Task(id=2, description="b"),
        Task(id=3, description="c"),
    ])
    assert [t.id for t in plan.roots()] == [1, 2, 3]
    for t in plan.tasks:
        assert plan.children_of(t.id) == []


def test_architect_prompt_documents_parent_id_field():
    """The architect must learn that `new_tasks` entries can carry an
    optional `parent_id`. Pin the prompt change so future edits don't
    silently drop it."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    assert '"parent_id"' in src
    assert "parent_id" in src


def test_apply_intent_to_plan_passes_parent_id():
    """When the architect's intent includes a new_tasks entry with
    parent_id, the orchestrator must pass it to Plan.insert."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    # The plumbing — passes parent_id keyword arg through
    assert "parent_id=parent_id" in src or 'parent_id=new.get("parent_id")' in src
