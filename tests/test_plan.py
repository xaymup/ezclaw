import pytest
from plan import Plan, Task, TASK_STATUSES


def make_plan(n=3, title="test plan"):
    return Plan(
        title=title,
        tasks=[Task(id=i, description=f"task {i}") for i in range(1, n + 1)],
    )


def test_task_defaults_to_pending():
    t = Task(id=1, description="hello")
    assert t.status == "pending"


def test_plan_get_task_returns_task_by_id():
    p = make_plan(3)
    assert p.get_task(2).description == "task 2"
    assert p.get_task(99) is None


def test_advance_to_in_progress_sets_current_task_id():
    p = make_plan(3)
    p.advance(2, "in_progress")
    assert p.get_task(2).status == "in_progress"
    assert p.current_task_id == 2


def test_advance_from_in_progress_to_done_clears_current_task_id():
    p = make_plan(3)
    p.advance(2, "in_progress")
    p.advance(2, "done")
    assert p.get_task(2).status == "done"
    assert p.current_task_id is None


def test_advance_to_failed_clears_current_task_id_if_matching():
    p = make_plan(3)
    p.advance(2, "in_progress")
    p.advance(2, "failed")
    assert p.current_task_id is None


def test_advance_to_skipped_clears_current_task_id_if_matching():
    p = make_plan(3)
    p.advance(2, "in_progress")
    p.advance(2, "skipped")
    assert p.current_task_id is None


def test_advance_with_invalid_status_is_noop():
    p = make_plan(3)
    p.advance(1, "bogus")
    assert p.get_task(1).status == "pending"


def test_advance_with_unknown_task_id_is_noop():
    p = make_plan(3)
    p.advance(99, "done")
    assert all(t.status == "pending" for t in p.tasks)


def test_insert_after_existing_task_assigns_new_id():
    p = make_plan(3)
    new_task = p.insert(2, "inserted")
    assert new_task.id == 4  # max existing id + 1
    assert new_task.description == "inserted"
    # Task should be inserted immediately after task 2
    ids_in_order = [t.id for t in p.tasks]
    assert ids_in_order == [1, 2, 4, 3]


def test_insert_after_unknown_id_appends_to_end():
    p = make_plan(3)
    new_task = p.insert(99, "appended")
    assert p.tasks[-1] is new_task


def test_is_complete_false_when_any_pending():
    p = make_plan(3)
    p.advance(1, "done")
    p.advance(2, "done")
    assert not p.is_complete()


def test_is_complete_true_when_all_done():
    p = make_plan(3)
    for i in range(1, 4):
        p.advance(i, "done")
    assert p.is_complete()


def test_is_complete_true_when_mix_of_done_and_skipped():
    p = make_plan(3)
    p.advance(1, "done")
    p.advance(2, "skipped")
    p.advance(3, "done")
    assert p.is_complete()


def test_is_complete_false_when_any_failed():
    p = make_plan(3)
    p.advance(1, "done")
    p.advance(2, "failed")
    p.advance(3, "done")
    assert not p.is_complete()


def test_progress_counts_done_and_skipped():
    p = make_plan(4)
    p.advance(1, "done")
    p.advance(2, "skipped")
    p.advance(3, "in_progress")
    done, total = p.progress()
    assert done == 2
    assert total == 4


def test_task_statuses_constant():
    assert "pending" in TASK_STATUSES
    assert "in_progress" in TASK_STATUSES
    assert "done" in TASK_STATUSES
    assert "failed" in TASK_STATUSES
    assert "skipped" in TASK_STATUSES


def test_advance_cannot_regress_done_task_to_in_progress():
    p = make_plan(3)
    p.advance(1, "done")
    p.advance(1, "in_progress")  # should be no-op
    assert p.get_task(1).status == "done"
    assert p.current_task_id is None


def test_advance_cannot_regress_skipped_task_to_in_progress():
    p = make_plan(3)
    p.advance(1, "skipped")
    p.advance(1, "in_progress")  # should be no-op
    assert p.get_task(1).status == "skipped"
    assert p.current_task_id is None
